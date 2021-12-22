package server;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class FederatedModel {

    public static int numOutputs = 6;
    private static final int HEIGHT = 32;
    private static final int WIDTH = 32;
    private static final int CHANNELS = 3;
    private static final int N_OUTCOMES = 6 ;
    private static final int N_SAMPLES_TESTING = 136;





    private static final String RESOURCES_FOLDER_PATH ="C:\\Users\\souvik\\Downloads\\dev\\FL3Tier\\client0\\res\\data_c\\test\\";

    public static MultiLayerNetwork model = null;
    private static final String serverModel = "res/serverModel/server_model.zip";


    public INDArray[][] fedavg(int layer, Map<Integer, Map<String, INDArray>> cache) throws IOException {
        int K = cache.size();
        System.out.println("The number of client is: " + K);
        System.out.println("Start conduct fedavg aggregation");

        INDArray[][] res = new INDArray[4][2];
        INDArray weightTmp_2;
        INDArray biasTmp_2;
        INDArray weightTmp_4;
        INDArray biasTmp_4;
        INDArray weightTmp_5;
        INDArray biasTmp_5;

        INDArray weight_2;
        INDArray bias_2;

        INDArray weight_4;
        INDArray bias_4;

        INDArray weight_5;
        INDArray bias_5;


        Map<String, INDArray> paramTable = cache.get(1);
//        System.out.println("The weight of 2 layer is " + paramTable.get(String.format("%d_W", 2)));

        weight_2 = paramTable.get(String.format("%d_W", 2));
        bias_2 = paramTable.get(String.format("%d_b", 2));
        weight_4 = paramTable.get(String.format("%d_W", 4));
        bias_4 = paramTable.get(String.format("%d_b", 4));
        weight_5 = paramTable.get(String.format("%d_W", 5));
        bias_5 = paramTable.get(String.format("%d_b", 5));

        for (int i = 2; i < K + 1; i++) {
            if (cache.containsKey(i)) {
                Map<String, INDArray> paramTableTmp = cache.get(i);
                weightTmp_2 = paramTableTmp.get(String.format("%d_W", 2));
                biasTmp_2 = paramTableTmp.get(String.format("%d_b", 2));
                weightTmp_4 = paramTableTmp.get(String.format("%d_W", 4));
                biasTmp_4 = paramTableTmp.get(String.format("%d_b", 4));
                weightTmp_5 = paramTableTmp.get(String.format("%d_W", 5));
                biasTmp_5 = paramTableTmp.get(String.format("%d_b", 5));
                weight_2 = weight_2.add(weightTmp_2);
                bias_2 = bias_2.add(biasTmp_2);
                weight_4 = weight_4.add(weightTmp_4);
                bias_4 = bias_4.add(biasTmp_4);
                weight_5 = weight_5.add(weightTmp_5);
                bias_5 = bias_5.add(biasTmp_5);
            }
        }

        weight_2 = weight_2.div(K);
        weight_4 = weight_4.div(K);
        weight_5 = weight_5.div(K);
        bias_2 = bias_2.div(K);
        bias_4 = bias_4.div(K);
        bias_5 = bias_5.div(K);

        model.setParam(String.format("%d_W", 2), weight_2);
        model.setParam(String.format("%d_b", 2), bias_2);
        model.setParam(String.format("%d_W", 4), weight_4);
        model.setParam(String.format("%d_b", 4), bias_4);
        model.setParam(String.format("%d_W", 5), weight_5);
        model.setParam(String.format("%d_b", 5), bias_5);

        res[0][0] = weight_2;
        res[0][1] = bias_2;
        res[1][0] = weight_4;
        res[1][1] = bias_4;
        res[2][0] = weight_5;
        res[2][1] = bias_5;

//        model.setParam(String.format("%d_W", layer), weight);

        System.out.println("\nWriting server model...");
        ModelSerializer.writeModel(model, serverModel, false);

//        evaluateModel();
//        cache.clear();
        DataSetIterator testDsi = getDataSetIterator(RESOURCES_FOLDER_PATH + "/testing", N_SAMPLES_TESTING);
        System.out.println("Evaluating Model...");
        Evaluation eval = model.evaluate(testDsi);
        System.out.println(eval.stats());

        return res;
    }



    private static DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
        try {
            File folder = new File(folderPath);
            File[] digitFolders = folder.listFiles();


            NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH); //28x28
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1); //translate image into seq of 0..1 input values

            INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT * WIDTH});
            INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

            int n = 0;
            //scan all 0 to 9 digit subfolders
            for (File digitFolder : digitFolders) {

//                System.out.println("the name is " + digitFolder.getName());
                int labelDigit = Integer.parseInt(digitFolder.getName());
                File[] imageFiles = digitFolder.listFiles();

                for (File imgFile : imageFiles) {
                    INDArray img = nativeImageLoader.asRowVector(imgFile);
                    scaler.transform(img);
                    input.putRow(n, img);
                    output.put(n, labelDigit, 1.0);
                    n++;
                }
            }//End of For-loop

            //Joining input and output matrices into a dataset
            DataSet dataSet = new DataSet(input, output);
            //Convert the dataset into a list
            List<DataSet> listDataSet = dataSet.asList();
            //Shuffle content of list randomly
            Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
            int batchSize = 50;

            //Build and return a dataset iterator
            DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);
            return dsi;
        } catch (Exception e) {
            System.out.println(e.getLocalizedMessage());
            return null;
        }
    } //End of DataIterator Method



    public static void delete(List<File> files) {
        System.out.println("Deleting files...");
        int len = files.size();
        for (int i = 0; i < len; i++) {
            files.get(i).delete();
        }
        System.out.println("Files deleted");
    }




    public void initModel() throws IOException {

        System.out.println("initing model...");
        int seed = 100;
       // double learningRate = 0.001;
        int round = 10;
        int numHiddenNodes = 1000;


       MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(Updater.ADAM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nIn(CHANNELS).nOut(32).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(16).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(32).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(128).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(numOutputs).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.AVG).build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(56).build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(56).build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(56).build())

                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numOutputs)
                        .dropOut(0.8)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
                .build();


        model = new MultiLayerNetwork(conf);
        model.conf();
        model.init();

        System.out.println("init model finish!\n");

        ModelSerializer.writeModel(model, serverModel, true);
        System.out.println("Write model to " + serverModel + " finish\n");

    }

}
