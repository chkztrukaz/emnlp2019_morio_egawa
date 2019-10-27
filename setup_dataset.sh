# Download ChangeMyView Dataset
wget http://katfuji.lab.tuat.ac.jp/nlp_datasets/dataset.zip
unzip dataset.zip
rm dataset.zip

cd dataset
# Download GloVe
download=glove.6B.zip
dirname=glove.6B
wget http://nlp.stanford.edu/data/glove.6B.zip -O ${download}
unzip ${download} -d ${dirname}
rm ${download}
mv ${dirname}/glove.6B.100d.txt ./glove.6B.100d
gzip glove.6B.100d
rm -r ${dirname}

cd ..