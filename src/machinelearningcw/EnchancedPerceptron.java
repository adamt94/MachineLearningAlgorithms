/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.util.Arrays;
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

    public double w[];// weights
    public int numberofiterations = 100; //stopping condition
    public double learning_rate = 1;  //learning rate
    public boolean setCrossvalidate = true;//set crossvalidation
    public boolean setStandardiseAttributes = true; // set stdandisation
    public boolean crossvalidate = false; //decides which algorithm to pick offline/online

    public double means[];// the means for each attribute 
    public double std[]; //the standard deviations for each attribute
    public int noOfiterations = 1;//stopping condition

    @Override
    public void buildClassifier(Instances i) throws Exception {
        w = new double[i.numAttributes() - 1];//weights
        Arrays.fill(w, 1);//sets all values to 1 should be radomised
        Instances temp = new Instances(i);
        if (setCrossvalidate) {
            crossvalidate = crossValidation(i);
        }
      //  System.out.println("using online: "+ crossvalidate);
        Arrays.fill(w, 1);//sets all values to 1 should be radomised

        // System.out.println(crossvalidate);
        //crossvalidate = false;
        if (crossvalidate == true) {
            calculateMeansAndSTDev(temp);
            if (setStandardiseAttributes) {
                temp = standardizeAtrrbutes(temp);
            }
            //  System.out.println(temp);
            double b = perceptron(temp);
         //   System.out.println("error count: " + b);

        }
        if (crossvalidate == false) {
            calculateMeansAndSTDev(temp);
            if (setStandardiseAttributes) {
                temp = standardizeAtrrbutes(temp);
            }
            double c = offlinePerceptron(temp);
        //    System.out.println("error count " + c);

        }
       /* for(int o =0; o<w.length;o++){
            System.out.println("WEIGHT "+ w[o]);
        }*/
//
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double y = 0;
        for (int i = 0; i < instnc.numAttributes() - 1; i++) {
            y += w[i] * (instnc.value(i));
        }
        
       

        return (y >= 0) ? 1 : 0;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public double offlinePerceptron(Instances ins) {
        double error_count = 0;//count the number of errors
        double changeinWeights[];
        for (int h = 0; h < numberofiterations; h++) {
            changeinWeights = new double[ins.numAttributes() - 1];
            //error_count = 0;
            for (Instance instance : ins) {
                double y = 0;
                for (int i = 0; i < ins.numAttributes() - 1; i++) {

                    y += w[i] * (instance.value(i));
                }
                double match;
                if (y >= 0) {
                    match = 1;
                } else {
                    match = 0;
                }
                double difference = instance.classValue() - match;

                for (int j = 0; j < ins.numAttributes() - 1; j++) {

                    changeinWeights[j] = changeinWeights[j] + (0.5 * learning_rate) * ((difference) * instance.value(j));

                }
                error_count += (difference * difference);

            }
          /*  for (int j = 0; j < ins.numAttributes() - 1; j++) {

                System.out.print("w[" + j + "]: " + changeinWeights[j] + "|");
            }
            System.out.println("");*/

            for (int j = 0; j < w.length; j++) {

                w[j] += changeinWeights[j];

            }

        }
        error_count = error_count/numberofiterations;// average error count
        return error_count;
    }

    public double perceptron(Instances ins) {
        double error_count = 0;//count the number of errors
        for (int h = 0; h < numberofiterations; h++)//stopping condition
        {
            error_count = 0;
            for (Instance instance : ins) {
                double y = 0;
                for (int i = 0; i < ins.numAttributes() - 1; i++) {
                    y += w[i] * (instance.value(i));
                }
                // System.out.println(y);

                double match;
                if (y >= 0) {
                    match = 1;
                } else {
                    match = 0;
                }
                double difference = instance.classValue() - match;

                //  System.out.println(match);
                for (int j = 0; j < ins.numAttributes() - 1; j++) {

                    w[j] = w[j] + 0.5 * learning_rate * ((difference) * instance.value(j));
                    // System.out.print(w[j] + ", ");

                }

                error_count += difference * difference;
            }
        }
        return error_count;

    }

    public void calculateMeansAndSTDev(Instances instances) {
          means = new double[instances.numAttributes() - 1];//intialize means
        std = new double[instances.numAttributes() - 1];//intialize stdevs
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
                double x = ((i.value(n) - means[n]) / std[n]);
               
                i.setValue(n, x);
                
            }
        }
        
        return ins;
    }
    
  
    public boolean crossValidation(Instances ins) throws Exception {
        //get the data
        Instances data = new Instances(ins);
        Instances train;// the new training data
        Instances test; // the new testing data

        int seed = 0;
        Random rand = new Random(seed);
        //randomize the data
        data.randomize(rand);

        //number of folds
        int folds = 10;
        int offlineErrors = 0;
        int onlineErrors = 0;

        for (int i = 0; i < folds; i++) {
            train = data.trainCV(folds, i);
            test = data.testCV(folds, i);

            //add the the total errors for each
            //offlineErrors += 
            offlinePerceptron(train);
            for (Instance inst : test) {
                if (classifyInstance(inst) != inst.classValue()) {
                    offlineErrors += 1;
                }

            }
            //reset w
            Arrays.fill(w, 1);
            perceptron(train);
            for (Instance inst : test) {
                if (classifyInstance(inst) != inst.classValue()) {
                    onlineErrors += 1;
                }
            }

        }
       System.out.println(" off: " + offlineErrors);
        System.out.println(" on: " + onlineErrors);
        //calculate the mean of the total errors
        offlineErrors = offlineErrors / folds;
        onlineErrors = onlineErrors / folds;
        // System.out.println(flag);
        return offlineErrors > onlineErrors;

    }

}
