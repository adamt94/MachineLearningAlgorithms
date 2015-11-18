/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

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
    static int numberofiterations = 10; //stopping condition
    static double learnig_rate = 0.5;  //learning rate
    boolean flag = true;

    @Override
    public void buildClassifier(Instances i) throws Exception {
        if (flag == true) {

        }

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

        for (int h = 0; h < numberofiterations; h++)  //stopping condition
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

            for (int j = 0; j > w.length; j++) {

                w[j] += changeinWeights[j];

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

    public void standardizeAtrrbutes(Instances instances) {

        double means[] = new double[instances.numAttributes() - 1];
        double std[] = new double[instances.numAttributes() - 1];
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            means[i] = instances.meanOrMode(i);//returns the mean of the instances

        }
        for (int j = 0; j < instances.numAttributes() - 1; j++) {
            Stats s = new Stats();
            for (int i = 0; i < instances.numInstances(); i++) {
                s.add(instances.get(i).value(j));//adds values to calc std
            }
            s.calculateDerived(); //calculates mean and stdDev

            means[j] = s.mean;
            std[j] = s.stdDev;
        }
        for (int j = 0; j < instances.numAttributes() - 1; j++) {
            for (Instance i : instances) {
                double x = i.value(j) - (means[j] / std[j]);
                i.setValue(j, x);

            }

        }
    }

}
