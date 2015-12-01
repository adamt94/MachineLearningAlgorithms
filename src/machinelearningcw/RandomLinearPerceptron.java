/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearningcw;

import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author ypf12pxu
 */
public class RandomLinearPerceptron implements Classifier {
       static EnchancedPerceptron ensemble[];

    @Override
    public void buildClassifier(Instances i) throws Exception {
        randomLinearPerceptron(i);
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
    
    public void randomLinearPerceptron(Instances ins) throws Exception {
        ensemble = new EnchancedPerceptron[500]; //default size 500
        for(int i=0; i<ensemble.length;i++)
        {
         ensemble[i].buildClassifier(ins);
        }

    }

    public static Instances randomAttributes(Instances ins) throws Exception {
        Instances newInstances; //new instances with the attributes removed
        int removeAttributes = (ins.numAttributes() - 1) - (int) Math.sqrt(ins.numAttributes() - 1);//get the amount of attributes to remove
        ArrayList<Integer> remove = new ArrayList<>();//arraylist to remove attributes positions
        Remove r = new Remove();//class to remove attributes

        
        for (int i = 0; i < ins.numAttributes() - 1; i++) {

            remove.add(i);
        }
        
        //shuffle the arraylist to get random numbers
        Collections.shuffle(remove);
        //remove attributes position numbers from the arraylist
        for (int i = 0; i < removeAttributes; i++) {
            remove.remove(i);
        }
        //add the arraylist to an array to use the method 
        int[] temp = new int[remove.size()];
        for (int i = 0; i < temp.length; i++) {
            temp[i] = remove.get(i);
        
        }
        //removes the attributes
        r.setAttributeIndicesArray(temp);
        //input the instances
        r.setInputFormat(ins);
        r.setInvertSelection(false);
        //new instance with the attributes removed
        newInstances = Filter.useFilter(ins, r);
        System.out.println(newInstances);

        return newInstances;

    }
    
}
