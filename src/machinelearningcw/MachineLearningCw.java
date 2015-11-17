/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcw;

import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author adam
 */
public class MachineLearningCw {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        Instances train = loadData("C:\\Users\\adam\\AppData\\Roaming\\Skype\\My Skype Received Files\\question1-train.arff");
        perceptronClassifier p = new perceptronClassifier();
        train.setClassIndex(train.numAttributes() - 1);
        
        System.out.println(train);
        p.buildClassifier(train);

    }

    public static Instances loadData(String path) {
        FileReader reader;
        Instances instances = null;
        try {
            reader = new FileReader(path);
            instances = new Instances(reader);
        } catch (Exception e) {
            System.out.println("Error: " + e);
        }
        return instances;
    }

}
