#!/bin/bash
#SBATCH --job-name=ng2nii
#SBATCH --output=logs/ng2nii_%j.out
#SBATCH --error=logs/ng2nii_%j.err
#SBATCH --time=00:60:00
#SBATCH --partition=minilab-cpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1


# indicate starting
echo "Starting slurm job..."


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate ng2nii-env1


# define size
size="96,96,96"


# brain AA1-PO-C-R45

# get coords, volume path, and output folder
coords1=(
"7819,7184,4847;7917,7280,4848"
"9236,7665,5428;9335,7761,5429"
"10180,6859,5141;10274,6953,5142"
"8285,8997,4711;8383,9094,4712"
"10541,9814,5257;10640,9913,5258"
"8808,10276,5361;8907,10373,5362"
"9235,10298,5390;9332,10395,5391"
"8688,11250,4965;8785,11348,4966"
"8559,12410,4747;8655,12506,4748"
"9471,12256,4704;9596,12353,4705"
)
vol_path1="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/TH_C-R45/Ex_647"
folder1="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/01_AA1-PO-C-R45"

# loop over coordinate sets
for i in "${!coords1[@]}"; do
    coord="${coords1[$i]}"
    suffix=$(printf "_p%02d" $((i + 1)))

    echo "Processing $coord with $suffix..."

    # run script
    python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
    --vol_path "$vol_path1" \
    --coord_input "$coord" \
    --folder "$folder1" \
    --size "$size" \
    --suffix "$suffix"
done

# --------------------------------------------------

# brain AE2-WF2a_A

# get coords, volume path, and output folder
coords2=(
"8739,7700,4193;8831,7792,4194"
"7484,8464,4041;7576,8565,4042"
"10237,6841,3764;10336,6934,3765"
"11246,8559,3915;11343,8656,3916"
"10113,7996,3851;10208,8093,3852"
"9209,8387,3851;9307,8483,3852"
"7745,9742,4330;7842,9837,4331"
"11092,9756,3904;11188,9851,3905"
"9192,11199,4267;9289,11297,4268"
"11281,11499,4435;11378,11596,4436"
)
vol_path2="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/CTIP2_3B-6/Ex_561"
folder2="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/02_AE2-WF2a_A"

# loop over coordinate sets
for i in "${!coords2[@]}"; do
    coord="${coords2[$i]}"
    suffix=$(printf "_p%02d" $((i + 1)))

    echo "Processing $coord with $suffix..."

    # run script
    python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
    --vol_path "$vol_path2" \
    --coord_input "$coord" \
    --folder "$folder2" \
    --size "$size" \
    --suffix "$suffix"
done

# --------------------------------------------------

# brain AJ12-LG1E-n_A

# get coords, volume path, and output folder
coords3=(
"27970,26270,3797;28066,26366,3798"
"24785,29569,4155;24876,29654,4156"
"26951,30272,4377;27044,30370,4378"
"26490,31792,4315;26582,31890,4316"
"27093,31972,2910;27189,32068,2911"
"28075,25868,3349;28174,25964,3350"
"30029,29191,2738;29918,29095,2739"
"29019,30347,2837;29118,30442,2838"
"28119,31935,3101;28214,32030,3102"
"28022,32371,2891;28123,32466,2892"
)
vol_path3="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/GFAP_1E-n/Ex_639"
folder3="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/03_AJ12-LG1E-n_A"

# loop over coordinate sets
for i in "${!coords3[@]}"; do
    coord="${coords3[$i]}"
    suffix=$(printf "_p%02d" $((i + 1)))

    echo "Processing $coord with $suffix..."

    # run script
    python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
    --vol_path "$vol_path3" \
    --coord_input "$coord" \
    --folder "$folder3" \
    --size "$size" \
    --suffix "$suffix"
done

# --------------------------------------------------

# brain AE2-WF2a_A

# get coords, volume path, and output folder
coords4=(
"26206,26423,4388;26302,26514,4389"
"27378,27117,4540;27473,27216,4541"
"28174,26552,4657;28272,26648,4658"
"26101,28395,4107;26196,28489,4108"
"25853,29518,3961;25948,29618,3962"
"27340,28473,3712;27436,28569,3713"
"29056,29274,3808;29152,29370,3809"
"26373,32564,4118;26477,32661,4119"
"27274,32782,3827;27380,32878,3828"
"27308,28438,4021;27405,28534,4022"
)
vol_path4="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/P75_AE2_2a/Ex_639"
folder4="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/04_AE2-WF2a_A"

# loop over coordinate sets
for i in "${!coords4[@]}"; do
    coord="${coords4[$i]}"
    suffix=$(printf "_p%02d" $((i + 1)))

    echo "Processing $coord with $suffix..."

    # run script
    python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
    --vol_path "$vol_path4" \
    --coord_input "$coord" \
    --folder "$folder4" \
    --size "$size" \
    --suffix "$suffix"
done

# --------------------------------------------------

# brain CH1-PCW1A_A

# get coords, volume path, and output folder
coords5=(
"26309,28218,4452;26408,28317,4453"
"27655,27690,3764;27754,27789,3765"
"26154,30358,3834;26254,30451,3835"
"26247,31245,3813;26347,31344,3814"
"27234,31220,3885;27334,31313,3886"
"27675,31064,3757;27774,31163,3758"
"26130,28661,3438;26220,28751,3439"
"26267,27007,3253;26365,27106,3254"
"26259,31284,3914;26356,31381,3915"
"27242,31293,3809;27338,31389,3810"
)
vol_path5="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/CH1_PCW1A_A/Ex_639"
folder5="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/05_CH1-PCW1A_A"

# loop over coordinate sets
for i in "${!coords5[@]}"; do
    coord="${coords5[$i]}"
    suffix=$(printf "_p%02d" $((i + 1)))

    echo "Processing $coord with $suffix..."

    # run script
    python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
    --vol_path "$vol_path5" \
    --coord_input "$coord" \
    --folder "$folder5" \
    --size "$size" \
    --suffix "$suffix"
done


# indicate completion
echo "Job complete!"










