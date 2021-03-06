# Adaptone-mixer

## Setup sur Windows
Sur Windows, il est possible d'utiliser le WSL pour compiler le code sous CLion ou simplement sous Ubuntu.

Il faut installer un Linux Subsystem sous Windows, [voir le tutoriel à cet effet](https://docs.microsoft.com/en-us/windows/wsl/install-win10). \
Je vous conseille de choisir Ubuntu.

Pour accéder à vos fichiers Windows sous Ubuntu, il suffit de `cd /mnt/[lettre_de_votre_disque]`.

Ouvrez Ubuntu.

Ensuite, sous Ubuntu, exécutez les commandes suivantes :
```
sudo apt-get install cmake gcc
sudo apt-get install libboost-all-dev
sudo apt-get install libssl-dev
sudo apt-get install alsa-utils
sudo apt-get install gfortran
sudo apt-get install libasound2-dev
```

Si vous avez un GPU Nvidia, vous devez télécharger le [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Installez CLion sur votre machine Windows, puis suivez [ce tutoriel](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-clion.html) afin de configurer votre WSL sous CLion.

Vous devriez pouvoir compiler le projet sous CLion. Pour ce faire, quand vous ouvrez le projet avec CLion, sélectionnez le dossier `source`. Le build devrait se faire automatiquement.

Si votre compilation ne fonctionne pas sous CLion, soyez sans craintes ! \
Dans Ubuntu, faites : 
```
cd Adaptone-mixer/source
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
cd build
make
```
Cette étape est très longue. Votre projet sera ensuite compilé.


## Setup sous Linux
Pour rouler le projet sur une machine Linux, il suffit d'installer les librairies suivantes :

```
sudo apt-get install cmake
sudo apt-get install libboost-all-dev
sudo apt-get install libssl-dev
sudo apt-get install alsa-utils
sudo apt-get install gfortran
sudo apt-get install libasound2-dev
```

## Setup des submodules
```shell
git submodule init
git submodule update
```

## Tests
Pour rouler les tests, créez un script `execute_tests.sh` dans le répertoire de votre choix (idéalement pas dans `Adaptone-mixer`) et mettez y ce contenu,en prenant soin de modifier les _paths_ si nécessaire :
```
#!/usr/bin/env sh
cd Adaptone-mixer/source/build/Mixer/test; ./MixerTests
cd ../../SignalProcessing/test; ./SignalProcessingTests
cd ../../Utils/test; ./UtilsTests
cd ../../Communication/test; ./CommunicationTests
```

Il suffit ensuite d'exécuter la commande `./execute_tests.sh`.


## Jetson TX2
Pour se connecter en `ssh` avec le Jetson TX2, il suffit d'utiliser ce script `copy_and_make_on_jetson.sh`, en prenant soin de modifier les _paths_ si nécessaire :
```
#!/usr/bin/env sh

sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "mkdir -p ~/Desktop/mixer"

sshpass -p "nvidia" rsync -au Adaptone-mixer/source/Communication/ nvidia@192.168.0.103:~/Desktop/mixer/Communication/
sshpass -p "nvidia" rsync -au Adaptone-mixer/source/Mixer/ nvidia@192.168.0.103:~/Desktop/mixer/Mixer/
sshpass -p "nvidia" rsync -au Adaptone-mixer/source/SignalProcessing/ nvidia@192.168.0.103:~/Desktop/mixer/SignalProcessing/
sshpass -p "nvidia" rsync -au Adaptone-mixer/source/Utils/ nvidia@192.168.0.103:~/Desktop/mixer/Utils/
sshpass -p "nvidia" rsync -au Adaptone-mixer/source/Uniformization/ nvidia@192.168.0.103:~/Desktop/mixer/Uniformization/

sshpass -p "nvidia" scp Adaptone-mixer/source/CMakeLists.txt nvidia@192.168.0.103:~/Desktop/mixer/CMakeLists.txt

sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "mkdir -p ~/Desktop/mixer/build"
sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build; cmake .. -DCMAKE_BUILD_TYPE=RELEASE"
sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build; make -j6"
```

Pour exécuter les tests en `ssh` sur le Jetson TX2, il suffit d'utiliser ce script :
```
#!/usr/bin/env sh
sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build/Mixer/test; ./MixerTests"
sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build/SignalProcessing/test; ./SignalProcessingTests"
sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build/Utils/test; ./UtilsTests"
sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build/Communication/test; ./CommunicationTests"
sshpass -p "nvidia" ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build/Uniformization/test; ./UniformizationTests"
```

Pour exécuter le code du mixeur, il suffit d'utiiser ce script `execute_mixer.sh`, en prenant soin de modifier les _paths_ si nécessaire :
```
#!/usr/bin/env sh
ssh -t nvidia@192.168.0.103 "cd ~/Desktop/mixer/build/Mixer; sudo ./Mixer"
```
