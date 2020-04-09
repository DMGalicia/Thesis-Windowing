/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author David
 */
public class AUC {
    public String getAUC(String window, String dataset) {
        //This method returns the Area Under the Curve AUC total, the AUC per class and the specificity and sensitivity per class in a string
        //The AUC is computed using decision trees as classifiers.
        double AUC = 0;
        String AUCperClass =  "";
        String SSperClass = "";
        try {
            Instances WD = ConverterUtils.DataSource.read(window);
            WD.setClassIndex(WD.numAttributes() - 1);
            Instances TD = ConverterUtils.DataSource.read(dataset);
            TD.setClassIndex(TD.numAttributes() - 1);
            J48 model = new J48();
            model.buildClassifier(WD);
            ArrayList<String> classList = new ArrayList<>();
            ArrayList<String> trueClass = new ArrayList<>();
            ArrayList<String> inducedClass = new ArrayList<>();
            for(int n = 0; n <TD.numInstances(); n++){
                Instance inst = TD.get(n);
                String clss = TD.classAttribute().value((int) inst.classValue());
                trueClass.add(clss);
                if (!classList.contains(clss)){
                    classList.add(clss);
                }
            }
            for (int m = 0; m<TD.numInstances(); m++){
                Instance inst = TD.get(m);
                String clss = TD.classAttribute().value((int) model.classifyInstance(inst));
                inducedClass.add(clss);
            }
            for (int k = 0; k<classList.size(); k++){
                int VP = 0;
                int VN = 0;
                int FP = 0;
                int FN = 0;
                int contClass = 0;
                for (int l = 0;  l<TD.numInstances(); l++){
                    if (trueClass.get(l).equals(classList.get(k))){
                        contClass ++;
                    }
                    if(trueClass.get(l).equals(classList.get(k)) && inducedClass.get(l).equals(classList.get(k))){
                        VP++;
                    }
                    if(!trueClass.get(l).equals(classList.get(k)) && !inducedClass.get(l).equals(classList.get(k))){
                    VN++;
                    }
                    if(trueClass.get(l).equals(classList.get(k)) && !inducedClass.get(l).equals(classList.get(k))){
                        FN++;
                    }
                    if(!trueClass.get(l).equals(classList.get(k)) && inducedClass.get(l).equals(classList.get(k))){
                        FP++; 
                    }
                }
                double Pr = ((double) contClass)/((double) trueClass.size());
                double Rc = ((double) VP)/((double) VP+FN);
                double Sp = ((double) VN)/((double) VN+FP);
                double Pa0 = (0.5*(Rc+Sp));
                AUCperClass += "C"+k+" = "+Pa0+";";
                SSperClass += "Rc(C"+k+") = "+Rc+" ;Pr(C"+k+") = "+Pr+"; ";
                double Pa = Pr*(0.5*(Rc+Sp));
                AUC += Pa;
            }
            AUC *= 100;
        } catch (Exception ex) {
            Logger.getLogger(AUC.class.getName()).log(Level.SEVERE, null, ex);
        }
        String res;
        res = AUC+","+AUCperClass+","+SSperClass;
        return res;
    }
}
