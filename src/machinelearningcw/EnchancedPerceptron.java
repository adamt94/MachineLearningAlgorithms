/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.util.Arrays;
import static machinelearningcw.perceptronClassifier.w;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;

/**
 *
 * @author ypf12pxu
 */
public class EnchancedPerceptron implements Classifier {

    static double w[];// weights
    static int numberofiterations = 1; //stopping condition
    static double learning_rate = 1;  //learning rate
    static boolean flag; //
    static double means[];// the means for each attribute 
    static double std[]; //the standard deviations for each attribute

    @Override
    public void buildClassifier(Instances i) throws Exception {
        w = new double[i.numAttributes() - 1];//weights
        means = new double[i.numAttributes() - 1];//intialize means
        std = new double[i.numAttributes() - 1];//intialize stdevs
        Arrays.fill(w, 1);//sets all values to 1 should be radomised
        Instances temp = new Instances(i);
        crossValidation(i);
        if (flag == true) {
            calculateMeansAndSTDev(temp);
            temp = standardizeAtrrbutes(temp);
            double b = offlinePerceptron(temp);
            System.out.println("error count: " + b);

        }
        if (flag == false) {
            perceptron(i);
        }
//
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        int y = 0;
        for (int i = 0; i < instnc.numAttributes() - 1; i++) {
            y += w[i] * (instnc.value(i));
        }

        return y;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static double offlinePerceptron(Instances ins) {
        int error_count = 0;//count the number of errors
        double changeinWeights[] = new double[ins.numAttributes() - 1];

        for (Instance instance : ins) {
            int y = 0;
            for (int i = 0; i < ins.numAttributes() - 1; i++) {

                y += w[i] * (instance.value(i));
            }
            int match;
            if (y >= 0) {
                match = 1;
            } else {
                match = 0;
            }
            double difference = instance.classValue() - match;
            for (int j = 0; j < ins.numAttributes() - 1; j++) {

                changeinWeights[j] = changeinWeights[j] + (0.5 * learning_rate) * ((difference) * instance.value(j));

            }
            error_count += difference * difference;

        }

        for (int j = 0; j < w.length; j++) {

            w[j] += changeinWeights[j];
         //   System.out.println(w[j]);

        }
        return error_count;
    }

    public static double perceptron(Instances ins) {
        int error_count = 0;//count the number of errors
        //   for (int h = 0; h < numberofiterations; h++)//stopping condition
        {
            for (Instance instance : ins) {
                int y = 0;
                for (int i = 0; i < ins.numAttributes() - 1; i++) {
                    y += w[i] * (instance.value(i));
                }
                // System.out.println(y);

                int match;
                if (y >= 0) {
                    match = 1;
                } else {
                    match = 0;
                }
                double difference = instance.classValue() - match;

                //  System.out.println(match);
                for (int j = 0; j < ins.numAttributes() - 1; j++) {

                    w[j] = w[j] + 0.5 * learning_rate * ((difference) * instance.value(j));

                }
                error_count += difference * difference;
            }
        }
        return error_count;

    }

    public static void calculateMeansAndSTDev(Instances instances) {
        for (int j = 0; j < instances.numAttributes() - 1; j++) {
            Stats s = new Stats();

            for (int i = 0; i < instances.numInstances(); i++) {
                s.add(instances.get(i).value(j));//adds values to calc std
            }
            s.calculateDerived(); //calculates mean and stdDev

            means[j] = s.mean;
            std[j] = s.stdDev;
        }

    }

    public Instances standardizeAtrrbutes(Instances ins) {

        for (Instance i : ins) {
            for (int n = 0; n < i.numAttributes() - 1; n++) {
                double x = (i.value(n) - means[n] / std[n]);
                i.setValue(n, x);
            }
        }

        return ins;
    }

    public static void crossValidation(Instances ins) {
        //get the data
        Instances data = new Instances(ins);
        Instances train;// the new training data
        Instances test; // the new testing data

        int seed = 0;
        Random rand = new Random(seed);
        //randomize the data
        data.randomize(rand);

        //number of folds
        int folds = 2;
        int offlineErrors = 0;
        int onlineErrors = 0;

        for (int i = 0; i < folds; i++) {
            train = data.trainCV(folds, i);
            test = data.testCV(folds, i);

            //add the the total errors for each
            offlineErrors += offlinePerceptron(train);
            onlineErrors += perceptron(train);

        }
        //calculate the mean of the total errors
        offlineErrors = offlineErrors / folds;
        onlineErrors = onlineErrors / folds;
        System.out.println(flag);
        flag = offlineErrors > onlineErrors;

    }

}
