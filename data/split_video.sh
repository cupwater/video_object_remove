partition=$1
input_video=$2

# split the input video into series of small videos
# startTime=0
# endTime=0
# vduration=20
# length=2160
# i=0
# while [ $endTime -le $length ]; do
#     #statements
#     i=$[$i+1]
#     endTime=$[$startTime+$vduration]
#     cat $startTime
#     srun -p $partition ffmpeg -i $input_video -ss $startTime -t $vduration -acodec copy -vcodec copy split_videos/$i.mp4 &
#     startTime=$[endTime]
# done

# extract frames from video
# i=1
# while [ $i -le 110 ]; do
#     #statements
#     mkdir split_frames/$i
#     srun -p $partition ffmpeg -i split_videos/${i}.mp4 -threads 1 -vf scale=-1:1900 -q:v 0 split_frames/$i/%05d.jpg &
#     i=$[$i+1]
# done

# get the 1-th frames of each splitted videos
#mkdir 1th_frames
i=1
while [ $i -le 110 ]; do
    #statements
    srun -p $partition cp split_frames/$i/00001.jpg 1th_frames/${i}_00001.jpg &
    i=$[$i+1]
done

