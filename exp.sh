#python2 main.py --run_dir result/ --cuda --model lstm --train data/askubuntu/train --eval data/android/ --bidir --d 100 --embedding word_vectors/glove.npz --max_epoch 50 --use_content --eval_use_content --criterion cosine

python main_domain.py --run_dir result-adv/ --cuda --model lstm --train data/askubuntu/train --eval data/android/ --bidir --d 100 --embedding word_vectors/glove.npz --max_epoch 50 --use_content --eval_use_content --criterion cosine --cross_train data/android/ --wasserstein
