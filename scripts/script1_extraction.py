import subprocess

# Liste des fichiers son à extraire
fileList = ['nn10b/nn10b_20220628_000000.wav',
            ]
outputFolder = 'wav'

# boucle for pour extraire tous les fichiers
for index, iFile in enumerate(fileList):
    command = 'aws s3 cp --no-sign-request s3://congo8khz-pnnn/recordings/wav/'+fileList[index]+ ' ./wav/'
    extract = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)