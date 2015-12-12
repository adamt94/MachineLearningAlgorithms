/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.io.FileReader;
import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.core.Capabilities;
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
    public static void main(String[] args) throws Exception {
        // Instances train = loadData("C:\\Users\\adam\\AppData\\Roaming\\Skype\\My Skype Received Files\\question1-train.arff");
        Instances train = loadData("\\\\ueahome4\\stusci5\\ypf12pxu\\data\\Documents\\Machine Learning\\cancer\\cancer-train.arff");
        Instances test = loadData("\\\\ueahome4\\stusci5\\ypf12pxu\\data\\Documents\\Machine Learning\\cancer\\cancer-test.arff");
        test.setClassIndex(test.numAttributes() - 1);
        train.setClassIndex(train.numAttributes() - 1);
       // System.out.println(train);
        compareAlgorithms(train, test);
        standardiseData(train, test);
        crossValidation(train, test);
      //  RandomLinearPerceptron p = new RandomLinearPerceptron();
     //   p.buildClassifier(train);
      //  System.out.println(getAccuracy(p,train));
        
        // perceptron(train, test);
        //  EnchancedPerceptron(train, test);
        //   RandomLinearPerceptron(train,test);
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

    public static double getAccuracy(Classifier s,Instances instances) throws Exception {
        int numFolds = 10;
        EvaluationUtils eval = new EvaluationUtils();
        ArrayList<Prediction> preds
                = eval.getCVPredictions(s, instances, numFolds);
        int correct = 0;
        int total = 0;
        for (Prediction pred : preds) {
            if (pred.predicted() == pred.actual()) {
                correct++;
            }
            total++;
        }
        double acc = (double) correct / total;
        return acc;
    }

    public static void compareAlgorithms(Instances train, Instances test) throws Exception {
        System.out.println("Compare Algorithms");
        EnchancedPerceptron ep = new EnchancedPerceptron();
        ep.crossvalidate = false;// set to choice offline
        ep.numberofiterations = 100;
        ep.setCrossvalidate = false;
        ep.setStandardiseAttributes = false;
   
        ep.buildClassifier(train);
        double errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - ep.classifyInstance(i)) * (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }

        System.out.println("original errors: " + errors);
        double per = (test.numInstances() - errors) / test.numInstances() * 100;
        System.out.println("Offline Accuracy: " + per);
        errors = 0;

        perceptronClassifier p = new perceptronClassifier();
        p.buildClassifier(train);

        for (Instance i : test) {

            errors += (i.classValue() - p.classifyInstance(i)) * (i.classValue() - p.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        per = (test.numInstances() - errors) / test.numInstances() * 100;
        System.out.println("Online Accuracy: " + per);
    }

    public static void standardiseData(Instances train, Instances test) throws Exception {
        System.out.println("\n" + "Compare Algorithms with standiasation");
        EnchancedPerceptron ep = new EnchancedPerceptron();
        ep.crossvalidate = false;// set to choice offline
        ep.numberofiterations = 100;
        ep.setCrossvalidate = false;
        ep.setStandardiseAttributes = true;//std attributes
        //standardise the test instances 
        Instances test2 = new Instances(test);

        ep.buildClassifier(train);
        ep.standardizeAtrrbutes(test2);
        double per = 0;
        double errors = 0;
        for (Instance i : test2) {

            errors += (i.classValue() - ep.classifyInstance(i)) * (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        per = (test2.numInstances() - errors) / test2.numInstances() * 100;
        System.out.println("offline Accuracy: " + per);

        ep.crossvalidate = true;
        ep.buildClassifier(train);
        errors = 0;
        for (Instance i : test2) {

            errors += (i.classValue() - ep.classifyInstance(i)) * (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        per = (test2.numInstances() - errors) / test2.numInstances() * 100;
        System.out.println("Online Accuracy: " + per);
    }

    public static void crossValidation(Instances train, Instances test) throws Exception {
        System.out.println("\n" + "CrossValidation");
        EnchancedPerceptron ep = new EnchancedPerceptron();
        ep.numberofiterations = 100;
        ep.setCrossvalidate = true;
        ep.setStandardiseAttributes = false;//std attributes

        ep.buildClassifier(train);
        double per = 0;
        double errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - ep.classifyInstance(i)) * (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        per = (test.numInstances() - errors) / test.numInstances() * 100;
        System.out.println("Online Accuracy: " + per);
    }

    public static void perceptron(Instances train, Instances test) throws Exception {
        perceptronClassifier p = new perceptronClassifier();
        test.setClassIndex(test.numAttributes() - 1);
        train.setClassIndex(train.numAttributes() - 1);
        p.buildClassifier(train);

        double errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - p.classifyInstance(i)) * (i.classValue() - p.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        double per = (test.numInstances() - errors) / test.numInstances() * 100;
        System.out.println("Accuracy: " + per);

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

    public static void RandomLinearPerceptron(Instances train, Instances test) throws Exception {
        RandomLinearPerceptron rlp = new RandomLinearPerceptron();
        EnchancedPerceptron p = new EnchancedPerceptron();
        rlp.buildClassifier(train);

        p.standardizeAtrrbutes(test);
        double errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - rlp.classifyInstance(i)) * (i.classValue() - rlp.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        double per = (test.numInstances() - errors) / test.numInstances() * 100;
        System.out.println("Accuracy: " + per);

    }

}
