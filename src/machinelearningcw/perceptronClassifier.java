/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author adam
 */
public class perceptronClassifier extends AbstractClassifier {

    double w[];// weights
    int numberofiterations = 100; //stopping condition
    double learning_rate = 01;  //learning rate

    @Override
    public void buildClassifier(Instances i) throws Exception {
        w = new double[i.numAttributes() - 1];//weights
        Arrays.fill(w, 1);//sets all values to 1 should be radomised
        double b = perceptron(i);
        // System.out.println(w);
       // System.out.println("errors: " + b);
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double y = 0;
        for (int i = 0; i < instnc.numAttributes() - 1; i++) {
            y += (w[i] * (instnc.value(i)));

        }

        return (y >= 0) ? 1 : 0;
    }

    public double perceptron(Instances ins) {
        double error_count = 0;//count the number of errors
        for (int h = 0; h < numberofiterations; h++) {
            error_count = 0;
            for (Instance instance : ins) {
                double y = 0;
                for (int i = 0; i < ins.numAttributes() - 1; i++) {
                    //   System.out.println("Y: "+ w[i]+"  "+instance.value(i));
                    y += (w[i] * (instance.value(i)));
                }
                
                //   System.out.println(y);

                double match;
                if (y >= 0) {
                    match = 1;
                } else {
                    match = 0;
                }

                double difference = instance.classValue() - match;

                //  System.out.println("class value: " + instance.classValue() 
                //+ "  "+match+"  " + difference + "difference");
                //System.out.println(match);
                for (int j = 0; j < ins.numAttributes() - 1; j++) {
                
                    //   System.out.println("w[" + j + "] = " + w[j] + " 
                    //+ " + "0.5 " + " * " + learning_rate + " * " + difference
                    //+ " * " + instance.value(j));
                    w[j] = w[j] + (0.5 * learning_rate * 
                            difference * instance.value(j));
                   
                    /*   System.out.println("w[" + j + "] = " + w[j] + " +
                            " + "0.5 " + " * " + learning_rate + " * " 
                            + difference + " * " + instance.value(j));*/

                }
                error_count += (difference * difference);
            }
        }
        //System.out.println(match);
        for (int j = 0; j < ins.numAttributes() - 1; j++) {

            //  System.out.print("w[" + j + "]: " + w[j] + "|");
        }

        //  System.out.println("");
        return error_count;

    }
}
