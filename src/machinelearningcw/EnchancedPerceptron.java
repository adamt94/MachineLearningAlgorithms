/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.util.Arrays;
import static machinelearningcw.perceptronClassifier.numberofiterations;
import static machinelearningcw.perceptronClassifier.w;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
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
    static double learnig_rate = 0.5;  //learning rate
    boolean flag = true;
    static double means[];// the means for each attribute 
    static double std[]; //the standard deviations for each attribute

    @Override
    public void buildClassifier(Instances i) throws Exception {
        w = new double[i.numAttributes() - 1];//weights
        means = new double[i.numAttributes() - 1];//intialize means
        std = new double[i.numAttributes() - 1];//intialize stdevs
        Arrays.fill(w, 1);//sets all values to 1 should be radomised
        if (flag == true) {
            calculateMeansAndSTDev(i);
            standardizeAtrrbutes(0,i.get(0).value(0));
            offlinePerceptron(i);
          

        }
//
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public static void offlinePerceptron(Instances ins) {
        double changeinWeights[] = new double[ins.numAttributes() - 1];

        for (int h = 0; h < numberofiterations; h++) //stopping condition
        {
            for (Instance instance : ins) {
                int y = 0;
                for (int i = 0; i < ins.numAttributes() - 1; i++) {

                    y += w[i] * (instance.value(i));
                }
                int match = checkmatch(ins.get(0), y);
                for (int j = 0; j < ins.numAttributes() - 1; j++) {

                    changeinWeights[j] = changeinWeights[j] + learnig_rate * ((instance.classValue() - match) * instance.value(j));

                }

            }

            for (int j = 0; j < w.length; j++) {

                w[j] += changeinWeights[j];
                System.out.println(w[j]);

            }

        }

    }

    public static int checkmatch(Instance i, int y) {

        if (y > 0) {
            return 1;
        } else {
            return 0;
        }
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

    public double standardizeAtrrbutes(int j, double attribute) {

//        for (int i = 0; i < instances.numAttributes() - 1; i++) {
//            means[i] = instances.meanOrMode(i);//returns the mean of the instances
//
// 
        double x = (attribute - means[j]) / std[j];
        System.out.println(x);

        return x;
    }

}
