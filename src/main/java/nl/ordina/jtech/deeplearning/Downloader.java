package nl.ordina.jtech.deeplearning;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator.Set;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

public class Downloader {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Downloader.class);

    private static final int BATCH_SIZE = 256;
    private static final Set EMNIST_SET = EmnistDataSetIterator.Set.DIGITS;
    private static EmnistDataSetIterator emnistTrain, emnistTest;

    public static void main(String[] args) {
        try {
            emnistTrain = new EmnistDataSetIterator(EMNIST_SET, BATCH_SIZE, true);
            emnistTest = new EmnistDataSetIterator(EMNIST_SET, BATCH_SIZE, false);

            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(emnistTrain);
            scaler.fit(emnistTest);

            log.info("Size of emnistTrain: " + EmnistDataSetIterator.numExamplesTrain(EMNIST_SET));

            log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");

            //Import the VGG16 from the Model Zoo
            ZooModel zooModel = VGG16.builder().build();
            //Initialize the pretrained weights
            zooModel.initPretrained();

            log.info("\n\nDownload finished!.\n\n");

        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

}
