from __future__ import annotations
import torch
from learned.stirnet.training.merge_aware_crops import build_merge_aware_crop_manifest
from learned.stirnet.training.crops import CropBatch, CropSpec, prepare_crop_batch
from learned.stirnet.training.source_corruption import apply_source_instance_dropout
from learned.stirnet.model.utils.contingency import label_contingency


def _scene():
    gt = torch.zeros((1,8,40,40), dtype=torch.long)
    gt[0,2:6,4:10,4:10]=1; gt[0,2:6,12:18,4:10]=2; gt[0,2:6,4:10,14:20]=3; gt[0,2:6,12:18,14:20]=4
    current = gt.clone(); current[(gt==1)|(gt==2)] = 10; current[gt==3]=11; current[gt==4]=12
    return gt, current, torch.tensor([[2.0,.4,.4]])


def test_merge_crop_is_selected_first_and_all_cells_are_covered():
    gt,current,spacing=_scene()
    m=build_merge_aware_crop_manifest(gt,current_labels=current,spacing_um=spacing,crop_shape_zyx=(8,28,28),min_complete_cells=3,preferred_complete_cells=4,context_um=2.0)
    assert m.records[0][0].candidate_type=='merge'; assert 10 in m.records[0][0].merge_source_ids; assert {1,2}.issubset(m.records[0][0].merge_gt_ids)
    assert m.uncoverable_cell_ids==((),); assert m.uncoverable_merge_source_ids==((),)


def test_partial_cell_is_ignored_not_relabelled_background():
    gt=torch.zeros((1,6,12,12),dtype=torch.long); gt[0,1:5,2:5,2:5]=1; gt[0,1:5,7:11,7:11]=2
    batch={'spatial_inputs':torch.zeros((1,5,6,12,12)),'instance_labels':gt.clone(),'spacing_um':torch.ones((1,3)),'dref_um':torch.tensor([4.0])}
    spec=CropSpec(0,(slice(0,6),slice(0,9),slice(0,9)),(6,12,12),torch.zeros(3),'coverage',(1,),(2,),(),())
    crop=prepare_crop_batch(batch,gt,[spec],partial_ignore_margin_um=0.0); valid=crop.batch['supervision_valid_mask'][0].cpu(); local=crop.gt_labels[0]
    assert bool(valid[local==1].all()); assert not bool(valid[local==2].any()); assert int((local==2).sum())>0


def test_contingency_does_not_count_ignored_partial_as_false_positive():
    pred=torch.tensor([[[1,1,2,2]]]); gt=torch.tensor([[[1,1,0,0]]]); valid=torch.tensor([[[True,True,False,False]]])
    assert set(label_contingency(pred,gt).row_ids.tolist())=={1,2}
    masked=label_contingency(pred,gt,valid_mask=valid); assert masked.row_ids.tolist()==[1]; assert masked.row_counts.tolist()==[2]


def test_source_dropout_keeps_raw_and_gt_and_skips_merge_crop():
    gt=torch.zeros((1,6,12,12),dtype=torch.long); gt[0,1:5,2:5,2:5]=1; gt[0,1:5,7:10,7:10]=2; current=gt.clone()
    spatial=torch.zeros((1,5,6,12,12)); spatial[:,0]=torch.rand((1,6,12,12)); spatial[:,1]=(current>0).float(); spatial[:,2]=spatial[:,1]*.5; spatial[:,4]=spatial[:,1]
    spec=CropSpec(0,(slice(0,6),slice(0,12),slice(0,12)),(6,12,12),torch.zeros(3),'coverage',(1,2),(),(),())
    crop=CropBatch({'spatial_inputs':spatial.clone(),'instance_labels':current.clone(),'supervision_valid_mask':torch.ones_like(gt,dtype=torch.bool)},gt.clone(),None,[spec]); raw=spatial[:,0].clone()
    changed=apply_source_instance_dropout(crop,probability=1.0,max_instances=1,seed=7); dropped=changed.batch['source_dropout_ids'][0]; assert len(dropped)==1
    mask=current[0]==dropped[0]; assert not bool((changed.batch['instance_labels'][0][mask]>0).any()); assert not bool((changed.batch['spatial_inputs'][0,1][mask]>0).any()); assert torch.equal(changed.batch['spatial_inputs'][:,0],raw); assert torch.equal(changed.gt_labels,gt)
    mspec=CropSpec(0,spec.slices_zyx,spec.full_shape_zyx,spec.center_shift_um,'merge',(1,2),(),(),(1,)); mcrop=CropBatch(crop.batch,crop.gt_labels,None,[mspec])
    unchanged=apply_source_instance_dropout(mcrop,probability=1.0,max_instances=1,seed=7); assert unchanged.batch['source_dropout_ids']==((),); assert torch.equal(unchanged.batch['instance_labels'],current)
