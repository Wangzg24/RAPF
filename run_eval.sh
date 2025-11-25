LR=1e-5
# ModelType=proto_yuanwen
ModelType=matpn_tri
#ModelType=pinjiechaxun
#ModelType=distillation
#ModelType=vae
#ModelType=xiangguanxing
# ModelType=proto
# ModelType=pair
#ModelType=gnn_get
#nodropPrototype-dropRelation-lr-1e-5
#dropPrototype-nodropRelation-lr-2e-5
#nodropPrototype-nodropRelation-lr-1e-5
#acl-camera-ready-$N-$K.pth.tar
N=5
K=1
# Dataset=test_fewrel
Dataset=val_pubmed_new

python test_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --model $ModelType --encoder bert --hidden_size 768 --val_step 2000 --test $Dataset \
    --batch_size 2 --only_test \
    --load_ckpt checkpoint/$ModelType/$N-$K-$LR-new_ld.pth.tar \
    --pretrain_ckpt pretrain \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model bert