#!/bin/bash

# infer images
python main.py --infer=model/plusai_v3_model.ckpt-29999  --log_dir=log --outpath=log/plusai_0415_ub3_left_prob_v3 --test_dir=/home/dijiang/Work/data/bag_extracted/image/left/20170415_urban_3_resize/img.txt --save_image=True --save_type=prob --save_dim=1

python data_utils.py -f probmask --maskpath log/plusai_0415_ub3_left_prob_v3 --inpath /home/dijiang/Work/data/bag_extracted/image/left/20170415_urban_3_resize --outpath log/plusai_0415_ub3_left_finalprob_v3
