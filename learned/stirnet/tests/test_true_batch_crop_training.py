from __future__ import annotations

import torch
import torch.nn.functional as F

from learned.stirnet import StirNet
from learned.stirnet.model.partition.rag import RAGCriterion, RAGTargets
from learned.stirnet.model.types import RAGState
from learned.stirnet.training.merge_aware_crops import (
    build_merge_aware_crop_manifest,
    sample_merge_aware_crop_specs,
)
from learned.stirnet.training.trainer import Trainer
from .conftest import fixed_stage_training, small_model_config, synthetic_batch


def _scene():
    gt = torch.zeros((1, 8, 48, 48), dtype=torch.long)
    gt[0,2:6,4:10,4:10]=1; gt[0,2:6,12:18,4:10]=2
    gt[0,2:6,4:10,18:24]=3; gt[0,2:6,12:18,18:24]=4
    gt[0,2:6,28:34,4:10]=5; gt[0,2:6,36:42,4:10]=6
    current=gt.clone(); current[(gt==1)|(gt==2)]=10; current[(gt==3)|(gt==4)]=11
    current[gt==5]=12; current[gt==6]=13
    return gt,current,torch.tensor([[2.0,.4,.4]])


def test_merge_aware_true_batch_is_deterministic_and_balanced():
    gt,current,spacing=_scene()
    manifest=build_merge_aware_crop_manifest(gt,current_labels=current,spacing_um=spacing,
        crop_shape_zyx=(8,28,28),min_complete_cells=2,preferred_complete_cells=3,views_per_cell=2,context_um=1.0)
    kw=dict(crops_per_step=1,global_step=0,crop_batch_size=4,merge_fraction=.5)
    a=sample_merge_aware_crop_specs(gt,spacing,manifest,**kw)
    b=sample_merge_aware_crop_specs(gt,spacing,manifest,**kw)
    assert len(a)==1 and len(a[0])==4
    key=lambda s: tuple((x.start,x.stop) for x in s.slices_zyx)
    assert [key(x) for x in a[0]]==[key(x) for x in b[0]]
    assert sum(bool(x.merge_source_ids) for x in a[0])==2
    assert sum(not bool(x.merge_source_ids) for x in a[0])==2
    assert len({key(x) for x in a[0]})==4


def test_early_crop_training_executes_one_true_batched_forward():
    config=small_model_config(); config.partition.rag_min_node_gt_support=0.0
    training=fixed_stage_training("spatial_partition")
    training.curriculum.refinement_crop_shape_zyx=(4,8,8)
    training.curriculum.refinement_crop_batch_size=2
    training.curriculum.refinement_crop_merge_fraction=.5
    training.curriculum.refinement_crops_per_step=1
    training.curriculum.geometry_bootstrap_crop_enabled=True
    training.curriculum.spatial_partition_crop_enabled=True
    trainer=Trainer(StirNet(config),training,device="cpu"); batches=[]
    handle=trainer.model.geometry_decoder.register_forward_pre_hook(lambda _,args:batches.append(int(args[0].shape[0])))
    try: metrics=trainer.train_step(synthetic_batch(temporal=False))
    finally: handle.remove()
    assert batches==[2]
    assert metrics["crop_count"]==2 and metrics["crop_true_batch_size"]==2
    assert metrics["phase_a_crop_effective_batch_size"]==2
    assert metrics["grad_geometry_spatial"]>0 and trainer.global_step==1


def _rag(logits):
    return RAGState(node_features=torch.zeros((4,1)),node_embeddings=torch.zeros((4,1)),
        node_batch=torch.tensor([0,0,1,1]),node_supervoxel_id=torch.tensor([1,2,1,2]),
        node_centroid_um=torch.zeros((4,3)),node_volume_voxels=torch.ones(4),
        edge_index=torch.tensor([[0,2,2,2],[1,3,3,3]]),edge_features=torch.zeros((4,1)),
        edge_embeddings=torch.zeros((4,1)),spatial_edge_logits=logits,
        edge_batch=torch.tensor([0,1,1,1]),supervoxel_labels=[],node_offsets=torch.tensor([0,2,4]))


def test_rag_batch_balancing_averages_crop_losses_not_edges():
    logits=torch.tensor([-2.,-2.,-2.,-2.]); target=torch.tensor([1.,0.,0.,0.]); rag=_rag(logits)
    targets=RAGTargets(target=target,valid=torch.ones(4,dtype=torch.bool),weight=torch.ones(4),
        node_purity=torch.ones(4),node_gt_support=torch.ones(4),dominant_gt=torch.ones(4,dtype=torch.long))
    criterion=RAGCriterion(); gt=torch.zeros((2,1,1,1),dtype=torch.long)
    balanced=criterion(rag,gt,targets=targets,batch_balanced=True)["rag_bce"]
    global_loss=criterion(rag,gt,targets=targets,batch_balanced=False)["rag_bce"]
    e0=F.binary_cross_entropy_with_logits(logits[:1],target[:1],pos_weight=torch.tensor(.5))
    e1=F.binary_cross_entropy_with_logits(logits[1:],target[1:],pos_weight=torch.tensor(3.))
    torch.testing.assert_close(balanced,.5*(e0+e1)); assert not torch.isclose(balanced,global_loss)
