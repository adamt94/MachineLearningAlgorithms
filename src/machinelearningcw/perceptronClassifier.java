/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author adam
 */
public class perceptronClassifier implements Classifier {
     static double w[];// weights
     static int numberofiterations = 10; //stopping condition
     static double n = 1;//learning rate
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        w = new double[i.numAttributes()-1];//weights
        Arrays.fill(w, 1);//sets all values to 1 should be radomised
        perceptron(i);
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

    public static void perceptron(Instances ins) {
        
        
        for (int h = 0; h < numberofiterations; h++)//stopping condition
        {
            for (Instance instance : ins) {
                int y = 0;
                for (int i = 0; i < ins.numAttributes() - 1; i++) {
                    y += w[i] * (instance.value(i));
                }
                System.out.println(y);
                int match = checkmatch(ins.get(0), y);
                System.out.println(match);
                for (int j = 0; j < ins.numAttributes() - 1; j++) {
                  
                    w[j] = w[j] + n * ((instance.classValue() - match) * instance.value(j));

                }

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
    
}
