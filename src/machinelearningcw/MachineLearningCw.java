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
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
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
    
    static ArrayList<String> fileNames;
    //file for writing the data into excel
    static FileWriter writer;
    
    public static void main(String[] args) throws Exception {
        //   Instances data = loadData("C:\\Users\\adam\\AppData\\Roaming\\Skype\\My Skype Received Files\\question1-train.arff");
        Instances data [] = getAllFiles();
      
        writer = new FileWriter("\\\\ueahome4\\stusci5\\ypf12pxu\\data\\Documents\\Machine Learning\\data.csv");
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
           
            System.out.println("==============="+fileNames.get(i)+"=============");
            writer.append(fileNames.get(i));
            writer.append(",");
            data[i].setClassIndex(data[i].numAttributes() - 1);

           
            compareAlgorithms(data[i]);
            standardiseData(data[i]);
            crossValidation(data[i]);
            writer.append("\n");
         //   RandomLinearPerceptron(data);
        }
        writer.flush();
        writer.close();

   
    }

    public static Instances[] getAllFiles() {
        File folder = new File("\\\\ueahome4\\stusci5\\ypf12pxu\\data\\Documents\\Machine Learning\\adamt94-machinelearning-da75565f2abe\\adamt94-machinelearning-da75565f2abe\\data_sets/");
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

    public static void compareAlgorithms(Instances data) throws Exception {
        System.out.println("Compare Algorithms");
        EnchancedPerceptron ep = new EnchancedPerceptron();
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
        double acc = ((double) correct / total) * 100;

        System.out.println("Offline Accuracy: " + acc);
        writer.append(acc+",");

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
        acc = ((double) correct / total) * 100;

        System.out.println("Online Accury: " + acc);
        writer.append(acc+",");
    }

    public static void standardiseData(Instances data) throws Exception {
        System.out.println("\n" + "Compare Algorithms with standiasation");
        EnchancedPerceptron ep = new EnchancedPerceptron();
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
        double acc = ((double) correct / total) * 100;

        System.out.println("Offline Accuracy: " + acc);
        writer.append(acc+",");
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
        acc = ((double) correct / total) * 100;

        System.out.println("Online Accury: " + acc);
        writer.append(acc+",");
    }

    public static void crossValidation(Instances data) throws Exception {
        System.out.println("\n" + "CrossValidation");
        EnchancedPerceptron ep = new EnchancedPerceptron();
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
        double acc = ((double) correct / total) * 100;
        if (ep.onlineoroffline) {
            System.out.println("picked: online " + "Accuracy " + acc);
              writer.append(acc+",");
        } else {
            System.out.println("picked: offline " + "Accuracy " + acc);
            writer.append(acc+",");
        }
    }

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
        double acc = ((double) correct / total) * 100;

        System.out.println(acc);

    }

    public static void EnchancedPerceptron(Instances train, Instances test) throws Exception {
        EnchancedPerceptron ep = new EnchancedPerceptron();
        ep.buildClassifier(train);
        if (ep.setStandardiseAttributes == true) {

            ep.standardizeAtrrbutes(test);
        }

        double errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - ep.classifyInstance(i)) * (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        double per = (test.numInstances() - errors) / test.numInstances() * 100;
        System.out.println("Accuracy: " + per);
    }

    public static void RandomLinearPerceptron(Instances data) throws Exception {
        System.out.println("\n ========RandomLinearPerceptron==========");
        RandomLinearPerceptron rlp = new RandomLinearPerceptron();
        //EnchancedPerceptron p = new EnchancedPerceptron();
        //  rlp.buildClassifier(data);

        //   p.standardizeAtrrbutes(test);
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
        double acc = ((double) correct / total) * 100;

        System.out.println("Random Accuracy: " + acc);
    }

}
