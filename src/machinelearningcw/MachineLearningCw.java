/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.io.FileReader;
import weka.classifiers.Classifier;
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
        Instances train = loadData("C:\\Users\\adam\\Documents\\Machine_Learning\\wins\\wins-train.arff");
        Instances test = loadData("C:\\Users\\adam\\Documents\\Machine_Learning\\wins\\wins-test.arff");
        test.setClassIndex(test.numAttributes() - 1);
        train.setClassIndex(train.numAttributes() - 1);

        compareAlgorithms(train, test);
        standardiseData(train, test);
        crossValidation(train, test);

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

        ep.buildClassifier(train);

        double per = 0;
        double errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - ep.classifyInstance(i)) * (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        per = (test.numInstances() - errors) / test.numInstances() * 100;
        System.out.println("offline Accuracy: " + per);

        ep.crossvalidate = true;
        ep.buildClassifier(train);
        errors = 0;
        for (Instance i : test) {

            errors += (i.classValue() - ep.classifyInstance(i)) * (i.classValue() - ep.classifyInstance(i));
            //System.out.println(errors);
        }
        System.out.println("original errors: " + errors);
        per = (test.numInstances() - errors) / test.numInstances() * 100;
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
