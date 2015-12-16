/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.core.Capabilities;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author adam
 */
public class MachineLearningCw {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    //array of all the dataset names
    static ArrayList<String> fileNames;
    //file for writing the data into excel
    static FileWriter writer;

    public static void main(String[] args) throws Exception {
       
        Instances data[] = getAllFiles();
        //writes the data to excel
        writer = new FileWriter("\\\\ueahome4\\stusci5\\ypf12pxu\\data\\Documents\\Machine Learning\\adamt94-machinelearning-da75565f2abe\\adamt94-machinelearning-da75565f2abe\\data.csv");
        writer.append("DataName");
        writer.append(",");//next column
        writer.append("Offline");
        writer.append(",");
        writer.append("Online");
        writer.append(",");
        writer.append("Offlinestd");
        writer.append(",");
        writer.append("Onlinestd");
        writer.append(",");
        writer.append("CrossValidation");
        writer.append(",");
        writer.append("Ensemble");
        writer.append(",");
        writer.append("WEKA1");
        writer.append(",");
        writer.append("WEKA2");
        writer.append("\n");//new row
        for (int i = 0; i < data.length; i++) {

            System.out.println("===============" + fileNames.get(i) + 
                    "=============");
            writer.append(fileNames.get(i));
            writer.append(",");
            data[i].setClassIndex(data[i].numAttributes() - 1);
            //1. Is one learning algorithm better than the other?
            //   compareAlgorithms(data[i]);

            /*2. Does standardising the data produce a 
             more accurate classifier? 
             You can test this on both learningalgorithms.*/
            //  standardiseData(data[i]);
            
            
            /*3. Does choosing the learning algorithm through 
             cross validation produce a more accurate classifier?*/
            //   crossValidation(data[i]);
            
            
            // 4. Does using an ensemble produce a more accurate classifier?
            //     ensemble(data[i]);
            
            
            /*5. Weka contains several related classifiers in the 
             package weka.classifiers.functions. 
             Comparetwo of your classifiers (including the ensemble) 
             to at least two of the following*/
            
            /*=======================================
                      Weka Classifiers
            =========================================*/
            
//            VotedPerceptron mp = new VotedPerceptron();
           // Logistic l = new Logistic();
          //  SimpleLogistic sl = new SimpleLogistic();
          //  MultilayerPerceptron mp = new MultilayerPerceptron();
          //  VotedPerceptron vp = new VotedPerceptron();
//            
//            int numFolds = 10;
//            EvaluationUtils eval = new EvaluationUtils();
//            ArrayList<Prediction> preds
//                    = eval.getCVPredictions(mp, data[i], numFolds);
//            int correct = 0;
//            int total = 0;
//            for (Prediction pred : preds) {
//                if (pred.predicted() == pred.actual()) {
//                    correct++;
//                }
//                total++;
//            }
//            double acc = ((double) correct / total);
//
//            System.out.println("Logistic Accuracy: " + acc);
//            writer.append(acc + ",");
            int j = data[i].numClasses();
            writer.append(j + ",");
            writer.append("\n");

        }
        
        
        /*=======================================================
         TIMING EXPIREMENT
         =========================================================
         */
        //create all the classifiers
        perceptronClassifier online = new perceptronClassifier();
        EnhancedLinearPerceptron offline = new EnhancedLinearPerceptron();
        EnhancedLinearPerceptron onlinestd = new EnhancedLinearPerceptron();
        onlinestd.setStandardiseAttributes = true;
        EnhancedLinearPerceptron offlinestd = new EnhancedLinearPerceptron();
        offlinestd.setStandardiseAttributes = true;
        EnhancedLinearPerceptron crossvalidate = new EnhancedLinearPerceptron();
        crossvalidate.setStandardiseAttributes = true;
        RandomLinearPerceptron random = new RandomLinearPerceptron();
        Logistic l = new Logistic();
        SimpleLogistic sl = new SimpleLogistic();
        MultilayerPerceptron mp = new MultilayerPerceptron();
        VotedPerceptron vp = new VotedPerceptron();
    //    timingExperiment(online, data);
        //  timingExperiment(offline, data);
        //timingExperiment(onlinestd, data);
        //timingExperiment(offlinestd, data);
        //timingExperiment(crossvalidate, data);
        timingExperiment(random, data);
        //timingExperiment(l, data);
        //timingExperiment(sl, data);
        //  timingExperiment(mp, data);
        // timingExperiment(vp, data);
        writer.flush();
        writer.close();

    }

    public static Instances[] getAllFiles() {
        File folder = new File("\\\\ueahome4\\stusci5\\ypf12pxu\\data\\Documents\\Machine Learning\\adamt94-machinelearning-da75565f2abe\\adamt94-machinelearning-da75565f2abe\\data_sets\\");
        File[] listOfFiles = folder.listFiles();
        fileNames = new ArrayList<>();
        Instances[] ins = new Instances[listOfFiles.length];
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                //get the names of all the datasets
                fileNames.add(listOfFiles[i].getName());
                //gets all the datasets paths and loads the data into instances
                ins[i] = loadData(listOfFiles[i].getPath());

            }
        }
        return ins;
    }

    public static Instances loadData(String path) {
        FileReader reader;
        Instances instances = null;
        try {
            reader = new FileReader(path);
            instances = new Instances(reader);
        } catch (Exception e) {
            System.out.println("Error: " + e);
        }
        return instances;
    }

    //1. Is one learning algorithm better than the other?
    public static void compareAlgorithms(Instances data) throws Exception {
        System.out.println("Compare Algorithms");
        EnhancedLinearPerceptron ep = new EnhancedLinearPerceptron();
        ep.onlineoroffline = false;// set to choice offline
        ep.numberofiterations = 100;
        ep.setCrossvalidate = false;
        ep.setStandardiseAttributes = false;

        int numFolds = 10;
        EvaluationUtils eval = new EvaluationUtils();
        ArrayList<Prediction> preds
                = eval.getCVPredictions(ep, data, numFolds);
        int correct = 0;
        int total = 0;
        for (Prediction pred : preds) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        double acc = ((double) correct / total);

        System.out.println("Offline Accuracy: " + acc);
        writer.append(acc + ",");

        perceptronClassifier p = new perceptronClassifier();
        numFolds = 10;
        eval = new EvaluationUtils();
        ArrayList<Prediction> preds2
                = eval.getCVPredictions(p, data, numFolds);
        correct = 0;
        total = 0;
        for (Prediction pred : preds2) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        acc = ((double) correct / total);

        System.out.println("Online Accury: " + acc);
        writer.append(acc + ",");
    }

    /*2. Does standardising the data produce a more accurate classifier? 
    You can test this on both learningalgorithms.*/
    public static void standardiseData(Instances data) throws Exception {
        System.out.println("\n" + "Compare Algorithms with standiasation");
        EnhancedLinearPerceptron ep = new EnhancedLinearPerceptron();
        ep.onlineoroffline = false;// set to choice offline
        ep.numberofiterations = 100;
        ep.setCrossvalidate = false;
        ep.setStandardiseAttributes = true;//std attributes
        //standardise the test instances 

        int numFolds = 10;
        EvaluationUtils eval = new EvaluationUtils();
        ArrayList<Prediction> preds
                = eval.getCVPredictions(ep, data, numFolds);
        int correct = 0;
        int total = 0;
        for (Prediction pred : preds) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        double acc = ((double) correct / total);

        System.out.println("Offline Accuracy: " + acc);
        writer.append(acc + ",");
        ep.onlineoroffline = true;

        numFolds = 10;
        eval = new EvaluationUtils();
        ArrayList<Prediction> preds2
                = eval.getCVPredictions(ep, data, numFolds);
        correct = 0;
        total = 0;
        for (Prediction pred : preds2) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        acc = ((double) correct / total);

        System.out.println("Online Accury: " + acc);
        writer.append(acc + ",");
    }

    /*3. Does choosing the learning algorithm through cross 
    validation produce a more accurate classifier?*/
    public static void crossValidation(Instances data) throws Exception {
        System.out.println("\n" + "CrossValidation");
        EnhancedLinearPerceptron ep = new EnhancedLinearPerceptron();
        ep.numberofiterations = 100;
        ep.setCrossvalidate = true;
        ep.setStandardiseAttributes = false;//std attributes

        int numFolds = 10;
        EvaluationUtils eval = new EvaluationUtils();
        ArrayList<Prediction> preds
                = eval.getCVPredictions(ep, data, numFolds);
        int correct = 0;
        int total = 0;
        for (Prediction pred : preds) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        double acc = ((double) correct / total);
        if (ep.onlineoroffline) {
            System.out.println("picked: online " + "Accuracy " + acc);
            writer.append(acc + ",");
        } else {
            System.out.println("picked: offline " + "Accuracy " + acc);
            writer.append(acc + ",");
        }
    }

    // 4. Does using an ensemble produce a more accurate classifier?
    public static void ensemble(Instances data) throws Exception {
        System.out.println("====ensemble method=====");
        RandomLinearPerceptron rlp = new RandomLinearPerceptron();
        int numFolds = 10;
        EvaluationUtils eval = new EvaluationUtils();
        ArrayList<Prediction> preds
                = eval.getCVPredictions(rlp, data, numFolds);
        int correct = 0;
        int total = 0;
        for (Prediction pred : preds) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        double acc = ((double) correct / total);
        writer.append(acc + ",");

        System.out.println("Random Accuracy: " + acc);

    }

    public static void wekaClassifiers() {
        Logistic l = new Logistic();
        SimpleLogistic sl = new SimpleLogistic();
        MultilayerPerceptron mp = new MultilayerPerceptron();
        VotedPerceptron vp = new VotedPerceptron();
    }

    public static void timingExperiment(Classifier s, Instances[] data) 
            throws Exception {

        /* get the biggest data set */
        Instances largestData = data[0];
        for (int i = 0; i < data.length; i++) {
            if (largestData.numInstances() < data[i].numInstances()) {
                largestData = data[i];
            }
        }
        for (int i = 1; i <= 7; i++) {
            int percent = i * 10;
            int train_size = (int) Math.round(largestData.numInstances()
                    * percent / 100);
            int testSize = largestData.numInstances() - train_size;
            Instances train = new Instances(largestData, 0, train_size);
            Instances test = new Instances(largestData, train_size, testSize);

            long t1 = System.currentTimeMillis();

            s.buildClassifier(train);
            for (Instance ins : test) {
                s.classifyInstance(ins);
            }

            long t2 = System.currentTimeMillis() - t1;
            //change to seconds

            System.out.println("TIME TAKEN " + i + ": " + t2);

        }
        System.out.println("\n");
    }
    /*============================================================== 
     Methods to test each Classifier
     ==============================================================*/

    public static void perceptron(Instances data) throws Exception {
        perceptronClassifier p = new perceptronClassifier();
        data.setClassIndex(data.numAttributes() - 1);

        int numFolds = 10;
        EvaluationUtils eval = new EvaluationUtils();
        ArrayList<Prediction> preds
                = eval.getCVPredictions(p, data, numFolds);
        int correct = 0;
        int total = 0;
        for (Prediction pred : preds) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        double acc = ((double) correct / total);

        System.out.println(acc);

    }

    public static void EnchancedPerceptron(Instances train, Instances test) 
            throws Exception {
        EnhancedLinearPerceptron ep = new EnhancedLinearPerceptron();
        ep.buildClassifier(train);
        if (ep.setStandardiseAttributes == true) {

            ep.standardizeAtrrbutes(test);
        }

        double errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - ep.classifyInstance(i)) *
                    (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        double per = (test.numInstances() - errors) / test.numInstances();
        System.out.println("Accuracy: " + per);
    }

    public static void RandomLinearPerceptron(Instances data) throws Exception {
        System.out.println("\n ========RandomLinearPerceptron==========");
        RandomLinearPerceptron rlp = new RandomLinearPerceptron();
        int numFolds = 10;
        EvaluationUtils eval = new EvaluationUtils();
        ArrayList<Prediction> preds
                = eval.getCVPredictions(rlp, data, numFolds);
        int correct = 0;
        int total = 0;
        for (Prediction pred : preds) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        double acc = ((double) correct / total);

        System.out.println("Random Accuracy: " + acc);
    }

}
