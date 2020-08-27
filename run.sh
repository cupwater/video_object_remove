video_list=(1 2 3 4 5 6 7 8 9)

# for i in ${video_list[@]}
# do 
#     MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
#     srun --mpi=pmi2 -p $1 -n1 --gres=gpu:4 --ntasks-per-node=4 --job-name=rmo \
#         python demo.py --data data/split_frames/$i
# done

for i in ${video_list[@]}
do 
    srun -p $1 -n1 -w SH-IDC1-10-5-36-240 --job-name=rmo \
        python demo.py --data data/split_frames/$i
done
