/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.util.Arrays;
import static machinelearningcw.EnchancedPerceptron.w;
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
    static double learning_rate = 1;  //learning rate

    @Override
    public void buildClassifier(Instances i) throws Exception {
        w = new double[i.numAttributes() - 1];//weights
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

   public static double perceptron(Instances ins) {
        int error_count = 0;//count the number of errors
  
        {
            for (Instance instance : ins) {
                int y = 0;
                for (int i = 0; i < ins.numAttributes() - 1; i++) {
                    y += w[i] * (instance.value(i));
                }
                System.out.println(y);

                int match;
                if (y >= 0) {
                    match = 1;
                } else {
                    match = 0;
                }
                double difference = instance.classValue() - match;

                System.out.println(match);
                for (int j = 0; j < ins.numAttributes() - 1; j++) {

                    w[j] = w[j] + 0.5 * learning_rate * ((difference) * instance.value(j));

                }
                error_count += difference * difference;
            }
        }
        return error_count;

    }
}
