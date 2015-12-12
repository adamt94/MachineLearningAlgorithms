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
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author ypf12pxu
 */
public class RandomLinearPerceptron implements Classifier {

    EnchancedPerceptron ensemble[];
    int attributesdeleted[][];

    @Override
    public void buildClassifier(Instances i) throws Exception {
        randomLinearPerceptron(i);
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        
        double[] result =distributionForInstance(instnc);
        if(result[0]>result[1])
        {
            return 0;
        }
        
        return 1;

    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
         Instance newInstance;
         double classify[] = new double[2];
       // 
        for(int n =0; n<instnc.numAttributes();n++)
        {
            
           // inst.setValue(n, instnc.value(n));
        }
        for (int i = 0; i < ensemble.length; i++) {
            newInstance = new DenseInstance(instnc);
            for(int j =0; j<attributesdeleted[i].length;j++)
            {
                newInstance.deleteAttributeAt(attributesdeleted[i][j]);
            }
           //  System.out.println("INSTANCE: "+newInstance);
             
            double result = ensemble[i].classifyInstance(newInstance);
            if(result == 0)
            {
                classify[0]+=1;
            }
            else{
                classify[1]+=1;
            }
        }
     //   System.out.println("0: "+ classify[0]+" 1: "+classify[1]);
        return classify;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public void randomLinearPerceptron(Instances ins) throws Exception {
        ensemble = new EnchancedPerceptron[500]; //default size 500
        //get the amount of attributes to remove
        int removeAttributes = (int) Math.sqrt(ins.numAttributes() - 1);
      
        //initialise the array for storing the attributes removed
        attributesdeleted = new int[ensemble.length][removeAttributes];
//        System.out.println("ENESEMALBE: "+attributesdeleted[0].length);
        for (int i = 0; i < ensemble.length; i++) {
            ensemble[i] = new EnchancedPerceptron();
          //  System.out.println("length: "+ensemble.length);
            //build the classifier with the random attributes
            
            ensemble[i].buildClassifier(randomAttributes(ins, i, removeAttributes));
          //  System.out.println("============== " + 0);
        }

    }

    public Instances randomAttributes(Instances ins, int position, int removeAttributes) throws Exception {
        Instances newInstances; //new instances with the attributes removed

        ArrayList<Integer> remove = new ArrayList<>();//arraylist to remove attributes positions
        Remove r = new Remove();//class to remove attributes
      
        for (int i = 0; i < ins.numAttributes() - 1; i++) {
            
            remove.add(i);
        }

        //shuffle the arraylist to get random numbers
        Collections.shuffle(remove);
        //remove attributes position numbers from the arraylist
       
        for (int i = 0; i < removeAttributes; i++) {
            
             //save the deleted attributes removed to array
            attributesdeleted[position][i] = remove.remove(i);
        }
      
       
        //add the arraylist to an array to use the remove method 
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
        //   System.out.println(newInstances);
        
        return newInstances;

    }

}
