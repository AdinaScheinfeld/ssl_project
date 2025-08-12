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

# vol path is the precomputed https path

# # brain 01: AA1-PO-C-R45

# # get coords, volume path, and output folder
# coords1=(
# "7819,7184,4847;7917,7280,4848"
# "9236,7665,5428;9335,7761,5429"
# "10180,6859,5141;10274,6953,5142"
# "8285,8997,4711;8383,9094,4712"
# "10541,9814,5257;10640,9913,5258"
# "8808,10276,5361;8907,10373,5362"
# "9235,10298,5390;9332,10395,5391"
# "8688,11250,4965;8785,11348,4966"
# "8559,12410,4747;8655,12506,4748"
# "9471,12256,4704;9596,12353,4705"
# )
# vol_path1="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/TH_C-R45/Ex_647"
# folder1="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/01_AA1-PO-C-R45"

# # loop over coordinate sets
# for i in "${!coords1[@]}"; do
#     coord="${coords1[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path1" \
#     --coord_input "$coord" \
#     --folder "$folder1" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 02: AE2-WF2a_A

# # get coords, volume path, and output folder
# coords2=(
# "8739,7700,4193;8831,7792,4194"
# "7484,8464,4041;7576,8565,4042"
# "10237,6841,3764;10336,6934,3765"
# "11246,8559,3915;11343,8656,3916"
# "10113,7996,3851;10208,8093,3852"
# "9209,8387,3851;9307,8483,3852"
# "7745,9742,4330;7842,9837,4331"
# "11092,9756,3904;11188,9851,3905"
# "9192,11199,4267;9289,11297,4268"
# "11281,11499,4435;11378,11596,4436"
# )
# vol_path2="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/CTIP2_3B-6/Ex_561"
# folder2="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/02_AE2-WF2a_A"

# # loop over coordinate sets
# for i in "${!coords2[@]}"; do
#     coord="${coords2[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path2" \
#     --coord_input "$coord" \
#     --folder "$folder2" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 03: AJ12-LG1E-n_A

# # get coords, volume path, and output folder
# coords3=(
# "27970,26270,3797;28066,26366,3798"
# "24785,29569,4155;24876,29654,4156"
# "26951,30272,4377;27044,30370,4378"
# "26490,31792,4315;26582,31890,4316"
# "27093,31972,2910;27189,32068,2911"
# "28075,25868,3349;28174,25964,3350"
# "30029,29191,2738;29918,29095,2739"
# "29019,30347,2837;29118,30442,2838"
# "28119,31935,3101;28214,32030,3102"
# "28022,32371,2891;28123,32466,2892"
# )
# vol_path3="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/GFAP_1E-n/Ex_639"
# folder3="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/03_AJ12-LG1E-n_A"

# # loop over coordinate sets
# for i in "${!coords3[@]}"; do
#     coord="${coords3[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path3" \
#     --coord_input "$coord" \
#     --folder "$folder3" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 04: AE2-WF2a_A

# # get coords, volume path, and output folder
# coords4=(
# "26206,26423,4388;26302,26514,4389"
# "27378,27117,4540;27473,27216,4541"
# "28174,26552,4657;28272,26648,4658"
# "26101,28395,4107;26196,28489,4108"
# "25853,29518,3961;25948,29618,3962"
# "27340,28473,3712;27436,28569,3713"
# "29056,29274,3808;29152,29370,3809"
# "26373,32564,4118;26477,32661,4119"
# "27274,32782,3827;27380,32878,3828"
# "27308,28438,4021;27405,28534,4022"
# )
# vol_path4="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/P75_AE2_2a/Ex_639"
# folder4="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/04_AE2-WF2a_A"

# # loop over coordinate sets
# for i in "${!coords4[@]}"; do
#     coord="${coords4[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path4" \
#     --coord_input "$coord" \
#     --folder "$folder4" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 05: CH1-PCW1A_A

# # get coords, volume path, and output folder
# coords5=(
# "26309,28218,4452;26408,28317,4453"
# "27655,27690,3764;27754,27789,3765"
# "26154,30358,3834;26254,30451,3835"
# "26247,31245,3813;26347,31344,3814"
# "27234,31220,3885;27334,31313,3886"
# "27675,31064,3757;27774,31163,3758"
# "26130,28661,3438;26220,28751,3439"
# "26267,27007,3253;26365,27106,3254"
# "26259,31284,3914;26356,31381,3915"
# "27242,31293,3809;27338,31389,3810"
# )
# vol_path5="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/CH1_PCW1A_A/Ex_639"
# folder5="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/05_CH1-PCW1A_A"

# # loop over coordinate sets
# for i in "${!coords5[@]}"; do
#     coord="${coords5[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path5" \
#     --coord_input "$coord" \
#     --folder "$folder5" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 06: AZ10-SR3B-6_A

# # get coords, volume path, and output folder
# coords6=(
# "9201,7988,3055;9294,8086,3056"
# "10418,8267,3271;10517,8365,3272"
# "7704,9502,3190;7804,9599,3191"
# "10911,9343,3142;11011,9439,3143"
# "9458,10299,3351;9545,10395,3352"
# "10703,10902,3601;10795,10999,3602"
# "7979,11913,3725;8078,12009,3726"
# "9780,12190,3873;9874,12284,3874"
# "10995,11151,3816;11091,11248,3817"
# "9890,12482,3950;9984,12577,3951"
# )
# vol_path6="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/AZ10_SR3B_6_A/Ex_488"
# folder6="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/06_AZ10-SR3B-6_A"

# # loop over coordinate sets
# for i in "${!coords6[@]}"; do
#     coord="${coords6[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path6" \
#     --coord_input "$coord" \
#     --folder "$folder6" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 07: AZ10-SR3B-6_A

# # get coords, volume path, and output folder
# coords7=(
# "10506,6510,3789;10602,6606,3790"
# "10169,7132,3906;10265,7228,3907"
# "11098,8432,3937;11198,8529,3938"
# "10419,9512,4029;10516,9608,4030"
# "10431,10181,4502;10529,10279,4503"
# "10902,9066,4882;11000,9164,4883"
# "9267,12207,5598;9374,12303,5599"
# "9365,6682,3743;9465,6781,3744"
# "9445,9912,4849;9545,10008,4850"
# "10366,10374,5192;10476,10470,5193"
# )
# vol_path7="https://wulab.cac.cornell.edu:8443/swift/v1/demo_datasets/AZ10_SR3B_6_A/Ex_647"
# folder7="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/07_AZ10-SR3B-6_A"

# # loop over coordinate sets
# for i in "${!coords7[@]}"; do
#     coord="${coords7[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path7" \
#     --coord_input "$coord" \
#     --folder "$folder7" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# brain 09: BH1-TM2B-f_A

# get coords, volume path, and output folder
coords09=(
"9806,8126,3420;9903,8223,3421"
"10307,8888,3551;10409,8984,3552"
"11242,9843,3784;11334,9935,3785"
"9769,11167,4110;9868,11266,4111"
"9776,12190,5647;9878,12287,5648"
"10360,10925,4756;10469,11024,4757"
"9237,9965,4605;9334,10062,4606"
"9753,8402,4642;9860,8500,4643"
"11390,10387,4534;11391,10480,4631"
"9390,11382,4388;9491,11479,4389"
)
vol_path09="https://wu-objstore-45d.med.cornell.edu/neuroglancer/BH1_TM2B_f_A/Ex_561"
folder09="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/09_BH1-TM2B-f_A"

# loop over coordinate sets
for i in "${!coords09[@]}"; do
    coord="${coords09[$i]}"
    suffix=$(printf "_p%02d" $((i + 1)))

    echo "Processing $coord with $suffix..."

    # run script
    python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
    --vol_path "$vol_path09" \
    --coord_input "$coord" \
    --folder "$folder09" \
    --size "$size" \
    --suffix "$suffix"
done

# --------------------------------------------------

# # brain 11: AA1-PO40-1-A

# # get coords, volume path, and output folder
# coords11=(
# "8807,8283,4216;8915,8381,4217"
# "8466,8628,4170;8569,8724,4171"
# "10076,8704,3942;10178,8801,3943"
# "8208,7170,4238;8305,7267,4239"
# "10835,8145,4238;10928,8242,4239"
# "10462,6461,4126;10556,6555,4127"
# "9114,11210,4111;9213,11309,4112"
# "7580,10501,3874;7678,10502,3972"
# "8644,6589,3590;8744,6683,3591"
# "10892,9462,4087;11002,9557,4088"
# )
# vol_path11="https://wu-objstore-45d.med.cornell.edu/neuroglancer/AA1_PO40_1_A/Ex_561"
# folder11="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/11_AA1-PO40-1-A"

# # loop over coordinate sets
# for i in "${!coords11[@]}"; do
#     coord="${coords11[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path11" \
#     --coord_input "$coord" \
#     --folder "$folder11" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 12: AA1-PO40-3-A

# # get coords, volume path, and output folder
# coords12=(
# "7968,8563,4410;8066,8661,4411"
# "10277,8880,4287;10384,8977,4288"
# "10056,9925,4472;10157,10020,4473"
# "9918,11022,4186;10013,11117,4187"
# "7386,11283,3988;7484,11378,3989"
# "11818,9923,4110;11914,10019,4111"
# "8237,10823,4372;8342,10918,4373"
# "10234,9515,4041;10337,9610,4042"
# "9178,12871,4258;9278,12968,4259"
# "9983,13398,4228;10079,13494,4229"
# )
# vol_path12="https://wu-objstore-45d.med.cornell.edu/neuroglancer/AA1_PO40_3_A/Ex_561"
# folder12="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/12_AA1-PO40-3-A"

# # loop over coordinate sets
# for i in "${!coords12[@]}"; do
#     coord="${coords12[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path12" \
#     --coord_input "$coord" \
#     --folder "$folder12" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# # brain 13: AA1-PO40-4-A

# # get coords, volume path, and output folder
# coords13=(
# "8509,10496,3798;8604,10595,3799"
# "8802,10203,3631;8900,10297,3632"
# "8250,11456,3843;8346,11552,3844"
# "10173,13081,3917;10274,13177,3918"
# "9748,13987,3876;9856,14085,3877"
# "9224,13832,3883;9329,13924,3884"
# "8751,13189,3873;8908,13288,3874"
# "8143,13575,3986;8238,13670,3987"
# "10090,9527,3356;10194,9624,3357"
# "8980,9238,3356;9083,9333,3357"
# )
# vol_path13="https://wu-objstore-45d.med.cornell.edu/neuroglancer/AA1_PO40_3_A/Ex_561"
# folder13="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/13_AA1-PO40-4-A"

# # loop over coordinate sets
# for i in "${!coords13[@]}"; do
#     coord="${coords13[$i]}"
#     suffix=$(printf "_p%02d" $((i + 1)))

#     echo "Processing $coord with $suffix..."

#     # run script
#     python /home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py \
#     --vol_path "$vol_path13" \
#     --coord_input "$coord" \
#     --folder "$folder13" \
#     --size "$size" \
#     --suffix "$suffix"
# done

# --------------------------------------------------

# indicate completion
echo "Job complete!"















