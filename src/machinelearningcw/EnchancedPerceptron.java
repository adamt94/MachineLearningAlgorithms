/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ypf12pxu
 */
public class EnchancedPerceptron implements Classifier {

    static double w[];// weights
    static int numberofiterations = 10; //stopping condition
    static double n = 0.5;//learning rate
    static boolean flag = true;

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

  public void standardizeAtrrbutes(Instances instances){
        double means[] = new double[instances.numAttributes()-1];
       for(int i =0; i<instances.numAttributes()-1;i++)
       {
           means[i] = instances.meanOrMode(i);
       }
    }

}
