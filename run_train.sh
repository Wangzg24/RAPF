LR=1e-5
#LR=1e-1
# ModelType=proto_yuanwen
#ModelType=pinjiechaxun
#ModelType=distillation
#ModelType=vae
#ModelType=cnnvae
#ModelType=gnn_get
#ModelType=xiangguanxing
ModelType=matpn_tri
#ModelType=proto
#ModelType=pair
#nodropPrototype-dropRelation-lr-1e-5
#dropPrototype-nodropRelation-lr-2e-5
#nodropPrototype-nodropRelation-lr-1e-5
#acl-camera-ready-$N-$K.pth.tar
N=5
K=1

python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --model $ModelType --encoder bert --hidden_size 768 --val_step 2000 --lr $LR \
    --pretrain_ckpt pretrain \
    --batch_size 2 --save_ckpt checkpoint/$ModelType/$N-$K-$LR-new_ld.pth.tar \
    --cat_entity_rep \
    --backend_model bert


#python train_demo.py \
#    --trainN $N --N $N --K $K --Q 1 --dot \
#    --model $ModelType --encoder cnn --hidden_size 230 --val_step 2000 --lr $LR \
#    --pretrain_ckpt pretrain \
#    --batch_size 2 --save_ckpt checkpoint/$ModelType/$N-$K-$LR-gate.pth.tar \
#    --cat_entity_rep \
#    --backend_model cnn