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
    public static int batchSize = 16;
    private static final int HEIGHT = 32;
    private static final int WIDTH = 32;
    private static final int CHANNELS = 3;
    private static final int N_OUTCOMES = 6 ;
    private static final int N_SAMPLES_TESTING = 507;

    int nSamples = 507;

    String filenameTrain = "/home/ubuntu/FL3Tier/server/res/trashnet/test/";
    private static final String RESOURCES_FOLDER_PATH ="/home/ubuntu/FL3Tier/server/res/trashnet/test/";


    public static MultiLayerNetwork model = null;
    private static final String serverModel = "res/serverModel/server_model.zip";


    public INDArray[][] fedavg(int layer, Map<Integer, Map<String, INDArray>> cache) throws IOException {
        int K = cache.size();
        System.out.println("The number of client is: " + K);
        System.out.println("Start conduct fedavg aggregation");

        INDArray[][] res = new INDArray[10][2];



        INDArray weightTmp_3;
        INDArray biasTmp_3;
        INDArray weightTmp_6;
        INDArray biasTmp_6;
        INDArray weightTmp_9;
        INDArray biasTmp_9;
        INDArray weightTmp_12;
        INDArray biasTmp_12;
        INDArray weightTmp_14;
        INDArray biasTmp_14;
        INDArray weightTmp_16;
        INDArray biasTmp_16;
        INDArray weightTmp_19;
        INDArray biasTmp_19;
        INDArray weightTmp_20;
        INDArray biasTmp_20;
        INDArray weightTmp_21;
        INDArray biasTmp_21;
        INDArray weightTmp_22;
        INDArray biasTmp_22;

        INDArray weight_3;
        INDArray bias_3;
        INDArray weight_6;
        INDArray bias_6;
        INDArray weight_9;
        INDArray bias_9;
        INDArray weight_12;
        INDArray bias_12;
        INDArray weight_14;
        INDArray bias_14;
        INDArray weight_16;
        INDArray bias_16;
        INDArray weight_19;
        INDArray bias_19;
        INDArray weight_20;
        INDArray bias_20;
        INDArray weight_21;
        INDArray bias_21;
        INDArray weight_22;
        INDArray bias_22;


        Map<String, INDArray> paramTable = cache.get(1);
//        System.out.println("The weight of 2 layer is " + paramTable.get(String.format("%d_W", 2)));

        weight_3 = paramTable.get(String.format("%d_W", 3));
        bias_3 = paramTable.get(String.format("%d_b", 3));
        weight_6 = paramTable.get(String.format("%d_W", 6));
        bias_6 = paramTable.get(String.format("%d_b", 6));
        weight_9 = paramTable.get(String.format("%d_W", 9));
        bias_9 = paramTable.get(String.format("%d_b", 9));
        weight_12 = paramTable.get(String.format("%d_W", 12));
        bias_12 = paramTable.get(String.format("%d_b", 12));
        weight_14 = paramTable.get(String.format("%d_W", 14));
        bias_14 = paramTable.get(String.format("%d_b", 14));
        weight_16 = paramTable.get(String.format("%d_W", 16));
        bias_16 = paramTable.get(String.format("%d_b", 16));
        weight_19 = paramTable.get(String.format("%d_W", 19));
        bias_19 = paramTable.get(String.format("%d_b", 19));
        weight_20 = paramTable.get(String.format("%d_W", 20));
        bias_20 = paramTable.get(String.format("%d_b", 20));
        weight_21 = paramTable.get(String.format("%d_W", 21));
        bias_21 = paramTable.get(String.format("%d_b", 21));
        weight_22 = paramTable.get(String.format("%d_W", 22));
        bias_22 = paramTable.get(String.format("%d_b", 22));

        for (int i = 2; i < K + 1; i++) {
            if (cache.containsKey(i)) {
                Map<String, INDArray> paramTableTmp = cache.get(i);

                weightTmp_3 = paramTableTmp.get(String.format("%d_W", 2));
                biasTmp_3 = paramTableTmp.get(String.format("%d_b", 2));
                weightTmp_6 = paramTableTmp.get(String.format("%d_W", 6));
                biasTmp_6 = paramTableTmp.get(String.format("%d_b", 6));
                weightTmp_9 = paramTableTmp.get(String.format("%d_W",9));
                biasTmp_9 = paramTableTmp.get(String.format("%d_b", 9));
                weightTmp_12 = paramTableTmp.get(String.format("%d_W", 12));
                biasTmp_12 = paramTableTmp.get(String.format("%d_b", 12));
                weightTmp_14 = paramTableTmp.get(String.format("%d_W",14));
                biasTmp_14 = paramTableTmp.get(String.format("%d_b", 14));
                weightTmp_16 = paramTableTmp.get(String.format("%d_W", 16));
                biasTmp_16 = paramTableTmp.get(String.format("%d_b", 16));
                weightTmp_19 = paramTableTmp.get(String.format("%d_W", 19));
                biasTmp_19 = paramTableTmp.get(String.format("%d_b", 19));
                weightTmp_20 = paramTableTmp.get(String.format("%d_W",20));
                biasTmp_20 = paramTableTmp.get(String.format("%d_b", 20));
                weightTmp_21 = paramTableTmp.get(String.format("%d_W", 21));
                biasTmp_21 = paramTableTmp.get(String.format("%d_b", 21));
                weightTmp_22 = paramTableTmp.get(String.format("%d_W",22));
                biasTmp_22 = paramTableTmp.get(String.format("%d_b", 22));
                weight_3 = weight_3.add(weightTmp_3);
                bias_3 = bias_3.add(biasTmp_3);
                weight_6 = weight_6.add(weightTmp_6);
                bias_6 = bias_6.add(biasTmp_6);
                weight_9 = weight_9.add(weightTmp_9);
                bias_9 = bias_9.add(biasTmp_9);
                weight_12 = weight_12.add(weightTmp_12);
                bias_12 = bias_12.add(biasTmp_12);
                weight_14 = weight_14.add(weightTmp_14);
                bias_14 = bias_14.add(biasTmp_14);
                weight_16 = weight_16.add(weightTmp_16);
                bias_16 = bias_16.add(biasTmp_16);
                weight_19 = weight_19.add(weightTmp_19);
                bias_19 = bias_19.add(biasTmp_19);
                weight_20 = weight_20.add(weightTmp_20);
                bias_20 = bias_20.add(biasTmp_20);
                weight_21 = weight_21.add(weightTmp_21);
                bias_21 = bias_21.add(biasTmp_21);
                weight_22 = weight_22.add(weightTmp_22);
                bias_22 = bias_22.add(biasTmp_22);
            }
        }

        weight_3 = weight_3.div(K);
        weight_6 = weight_6.div(K);
        weight_9 = weight_9.div(K);
        weight_12 = weight_12.div(K);
        weight_14 = weight_3.div(K);
        weight_16 = weight_6.div(K);
        weight_19 = weight_19.div(K);
        weight_20 = weight_20.div(K);
        weight_21 = weight_21.div(K);
        weight_22 = weight_22.div(K);

        bias_3 = bias_3.div(K);
        bias_6 = bias_6.div(K);
        bias_9 = bias_9.div(K);
        bias_12 = bias_12.div(K);
        bias_14 = bias_14.div(K);
        bias_16 = bias_16.div(K);
        bias_19 = bias_19.div(K);
        bias_20 = bias_20.div(K);
        bias_21 = bias_21.div(K);
        bias_22 = bias_22.div(K);


        model.setParam(String.format("%d_W", 3), weight_3);
        model.setParam(String.format("%d_b", 3), bias_3);
        model.setParam(String.format("%d_W", 6), weight_6);
        model.setParam(String.format("%d_b", 6), bias_6);

        model.setParam(String.format("%d_W", 9), weight_9);
        model.setParam(String.format("%d_b", 9), bias_9);
        model.setParam(String.format("%d_W", 12), weight_12);
        model.setParam(String.format("%d_b", 12), bias_12);

        model.setParam(String.format("%d_W", 14), weight_14);
        model.setParam(String.format("%d_b", 14), bias_14);
        model.setParam(String.format("%d_W", 16), weight_16);
        model.setParam(String.format("%d_b", 16), bias_16);

        model.setParam(String.format("%d_W", 19), weight_19);
        model.setParam(String.format("%d_b", 19), bias_19);
        model.setParam(String.format("%d_W", 20), weight_20);
        model.setParam(String.format("%d_b", 20), bias_20);

        model.setParam(String.format("%d_W", 21), weight_21);
        model.setParam(String.format("%d_b", 21), bias_21);
        model.setParam(String.format("%d_W", 22), weight_22);
        model.setParam(String.format("%d_b", 22), bias_22);

        res[0][0] = weight_3;
        res[0][1] = bias_3;
        res[1][0] = weight_6;
        res[1][1] = bias_6;

        res[2][0] = weight_9;
        res[2][1] = bias_9;
        res[3][0] = weight_12;
        res[3][1] = bias_12;

        res[4][0] = weight_14;
        res[4][1] = bias_14;
        res[5][0] = weight_16;
        res[5][1] = bias_16;

        res[6][0] = weight_19;
        res[6][1] = bias_19;
        res[7][0] = weight_20;
        res[7][1] = bias_20;

        res[8][0] = weight_21;
        res[8][1] = bias_21;
        res[9][0] = weight_22;
        res[9][1] = bias_22;


        System.out.println("\nWriting server model...");
        ModelSerializer.writeModel(model, serverModel, false);

        DataSetIterator testDsi = getDataSetIterator(RESOURCES_FOLDER_PATH, N_SAMPLES_TESTING);
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


    // average weights over mobile devices' models
    public void AverageWeights(int layer, double alpha, int K) throws IOException {
        System.out.println("The number of client is: " + K);

        //original model
        Map<String, INDArray> paramTable = model.paramTable();
        System.out.println(paramTable);
        INDArray weight = paramTable.get(String.format("%d_W", layer));
        INDArray bias = paramTable.get(String.format("%d_b", layer));
        INDArray avgWeights = weight.mul(alpha);
        //System.out.println("the avgWeight is :\n" + avgWeights);
        INDArray avgBias = bias.mul(alpha);
        //System.out.println("the avgBias is :\n" + avgBias);

        // average weights over mobile devices' models
        System.out.println("\nAveraging weights...");

        MultiLayerNetwork transferred_model = null;
        for (int i = 1; i < K + 1; i++) {

            if (FileServer.cache.containsKey(i)) {
                System.out.println("enter cache");
                paramTable = FileServer.cache.get(i);
 
                weight = paramTable.get("weight");

                bias = paramTable.get("bias");

                avgWeights = avgWeights.add(weight.mul(1.0 - alpha).div(K));
                avgBias = avgBias.add(bias.mul(1.0 - alpha).div(K));
            }
        }

        model.setParam(String.format("%d_W", layer), avgWeights);
        model.setParam(String.format("%d_b", layer), avgBias);

        System.out.println("\nWriting server model...");
        ModelSerializer.writeModel(model, serverModel, false);
        System.out.println("\nWriting server model Finished...");
        evaluateModel();

        FileServer.cache.clear();
    }

    public static void delete(List<File> files) {
        System.out.println("Deleting files...");
        int len = files.size();
        for (int i = 0; i < len; i++) {
            files.get(i).delete();
        }
        System.out.println("Files deleted");
    }

    public  void evaluateModel() throws IOException {


        File folder = new File(filenameTrain);
        File[] digitFolders = folder.listFiles();

        NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scalar = new ImagePreProcessingScaler(0,1);
        INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT*WIDTH});
        INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

        int n = 0;
        for (File digitFolder: digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            File[] imageFiles = digitFolder.listFiles();

            for (File imgFile : imageFiles) {
                INDArray img = nativeImageLoader.asRowVector(imgFile);
                //INDArray img = nativeImageLoader.asMatrix(imgFile);
                scalar.transform(img);
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                n++;
            }
        }
        //Joining input and output matrices into a dataset
        DataSet dataSet = new DataSet(input, output);
        //Convert the dataset into a list
        List<DataSet> listDataSet = dataSet.asList();
        //Shuffle content of list randomly
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));

        //Build and return a dataset iterator
        DataSetIterator testDsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);


        System.out.print("Evaluating Model...");
        Evaluation eval = model.evaluate(testDsi);
        System.out.print(eval.stats());
    }


    public void initModel() throws IOException {

        System.out.println("initing model...");
        int seed = 100;
        int round = 5;
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
