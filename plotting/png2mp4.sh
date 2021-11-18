#!/bin/sh
# PNG encoding with ffmpeg
# Options:
#   -y                  Overwrite output
#   -f image2pipe       Input format
#   -vcodec png         Input codec
#   -r $3               Frame rate
#   -i -                Input files from cat command
#   -f mp4              Output format
#   -vcodec libx254     Output codec
#   -pix_fmt yuv420p    Output pixel format
#   -preset slower      Prefer slower encoding / better results
#   -crf 20             Constant rate factor (lower for better quality)
#   -vf "scale..."      Round to even size
#   $2                  Output file
#
#Example
# png2mp4 "potential_0/potential_0_fields*.png" output.mp4 30
cat $1* | ffmpeg3 \
    -y \
    -f image2pipe \
    -vcodec png \
    -r $3 \
    -i - \
    -f mp4 \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -preset slower \
    -crf 20 \
    -vf "scale=trunc(in_w/2)*2:trunc(in_h/2)*2" \
    $2
