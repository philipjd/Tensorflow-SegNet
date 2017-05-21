#!/bin/bash

src=$1
des=$2

# Step 1. Generate mask label & copy data
echo "Generating mask and copy data"
if [ ! -d $des/sample ]
then
    mkdir -p $des/sample
fi

if [ ! -d $des/label ]
then
    mkdir -p $des/label
fi

ls $src/*.json | xargs -n1 basename > $src/tmp
for f in `cat $src/tmp`
do
    echo $f
    fname=${f%.*}
    if [ ! -d $src/${fname}_json ]; then
        labelme_json_to_dataset $src/$f
        if [ $? != 0 ];then echo "Failed!"; exit -1; fi
    fi

    cp $src/$fname.png $des/sample/
    if [ $? != 0 ];then echo "Failed!"; exit -1; fi

    cp $src/${fname}_json/label.png $des/label/${fname}.png
    if [ $? != 0 ];then echo "Failed!"; exit -1; fi

done

rm $src/tmp


# Step 2. Resize
echo "Resizing images"
ls $des/sample/*.png | xargs -n1 basename > $des/sample/tmp.txt
ls $des/label/*.png | xargs -n1 basename > $des/label/tmp.txt
python data_utils.py -f resize --infile $des/sample/tmp.txt
if [ $? != 0 ];then echo "Failed!"; exit -1; fi
python data_utils.py -f resize --infile $des/label/tmp.txt
if [ $? != 0 ];then echo "Failed!"; exit -1; fi


# Step 3. Process mask label
echo "Preprocessing label images"
python data_utils.py -f road --infile $des/label/tmp.txt
if [ $? != 0 ];then echo "Failed!"; exit -1; fi
rm $des/sample/tmp.txt
rm $des/label/tmp.txt


# Step 4. Augmentation
echo "Augmentation"
python data_utils.py -f augment --inpath $des/sample/ --maskpath $des/label/
if [ $? != 0 ];then echo "Failed!"; exit -1; fi

# Step 5. Split training/validation set
echo "Split training/validation set"
python data_utils.py -f road_split --inpath $des
if [ $? != 0 ];then echo "Failed!"; exit -1; fi


# Step 5. Pack
echo "Packing"
cd $des/../
dir=`basename $des`
if [ $? != 0 ];then echo "Failed!"; exit -1; fi

tar cjf - $dir | split -b 100m - $dir.tar.bz2
