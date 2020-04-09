/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.util.Pair;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 *
 * @author David
 */
public class MDL {

    //This method returns a string with the number of train instances, number of nodes, number of leaves, bits to encode the tree, bits to encode the train data,
    //train MDL, bits to encode the test data, test MDL, and the test accuracy.
    //The MDL and accuracy are computed using decision trees as classifiers.
    //Data -> Induction/Train, Data2 -> Test
    public String getMDL(Instances data, Instances data2) {
        String result = "";
        try {
            int numAtrib = data.numAttributes() - 1;
            data.setClassIndex(numAtrib);
            data2.setClassIndex(numAtrib);
            int numClase = data.numClasses();
            J48 tree = new J48();
            tree.buildClassifier(data);
            int numHojas = (int) tree.measureNumLeaves();
            int numNodos = (int) tree.measureTreeSize() - numHojas;
            ArrayList<String> IdNodos = new ArrayList<>();
            ArrayList<ArrayList<String>> ClasesNodos = new ArrayList<>();
            ArrayList<String> defaultClaseNodo = new ArrayList<>();
            for (int i = 0; i < data.numInstances(); i++) {
                double arreglo[] = tree.getMembershipValues(data.get(i));
                for (int j = 0; j < arreglo.length; j++) {
                    if (arreglo[j] > 0.5) {
                        arreglo[j] = 1.0;
                    } else {
                        arreglo[j] = 0.0;
                    }
                }
                String ArrayNodo = Arrays.toString(arreglo);
                if (IdNodos.contains(ArrayNodo)) {
                    int index = IdNodos.indexOf(ArrayNodo);
                    ClasesNodos.get(index).add(data.get(i).toString(data.get(i).classAttribute(), (int) data.get(i).classValue()));
                } else {
                    IdNodos.add(ArrayNodo);
                    int index = IdNodos.indexOf(ArrayNodo);
                    ArrayList<String> NuevoArray = new ArrayList<>();
                    ClasesNodos.add(NuevoArray);
                    ClasesNodos.get(index).add(data.get(i).toString(data.get(i).classAttribute(), (int) data.get(i).classValue()));
                }
            }
            result += (data.numInstances()+","+numNodos+","+numHojas+",");
            double bitsTree = bitsTree(numNodos, numHojas, numAtrib, numClase);
            result += (bitsTree+",");
            double bitsExceptions = 0;
            for (int i = 0; i < ClasesNodos.size(); i++) {
                ArrayList<String> Nodo = ClasesNodos.get(i);
                Pair<Double, String> bitsPerNode = bitsExcepciones(Nodo, i);
                bitsExceptions += bitsPerNode.getKey();
                defaultClaseNodo.add(bitsPerNode.getValue());
            }
            result += (bitsExceptions+",");
            result += ((bitsTree + bitsExceptions)+",");
            for (int i = 0; i < ClasesNodos.size(); i++) {
                ClasesNodos.get(i).clear();
            }
            for (int i = 0; i < data2.numInstances(); i++) {
                double arreglo[] = tree.getMembershipValues(data2.get(i));
                for (int j = 0; j < arreglo.length; j++) {
                    if (arreglo[j] > 0.5) {
                        arreglo[j] = 1.0;
                    } else {
                        arreglo[j] = 0.0;
                    }
                }
                String ArrayNodo = Arrays.toString(arreglo);
                if (IdNodos.contains(ArrayNodo)) {
                    int index = IdNodos.indexOf(ArrayNodo);
                    ClasesNodos.get(index).add(data2.get(i).toString(data2.get(i).classAttribute(), (int) data2.get(i).classValue()));
                } else {
                }
            }
            bitsExceptions = 0;
            for (int i = 0; i < ClasesNodos.size(); i++) {
                ArrayList<String> Nodo = ClasesNodos.get(i);
                Double bitsPerNode = bitsExcepciones(Nodo, i, defaultClaseNodo.get(i));
                bitsExceptions += bitsPerNode;
            }
            result += (bitsExceptions+",");
            result += ((bitsTree + bitsExceptions)+",");
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(tree, data2);
            result += (eval.pctCorrect()+",");
        } catch (Exception ex) {
            Logger.getLogger(MdlAucDt.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }
    
        public static Double bitsTree(double nNodes, double nLeaves, double nAttri, double nClass){
        double bits = nNodes*(1+(Math.log(nAttri)/Math.log(2.0)))+nLeaves*(1+(Math.log(nClass)/Math.log(2.0)));
        return bits;
    }
    
    public static Pair<Double,String> bitsExcepciones(ArrayList<String> n,int j){
        int longitud = n.size();
        HashSet<String> uniqueClass = new HashSet<>(n);
        int numClasses = uniqueClass.size();
        double totalBits = 0;
        String defaultClass = null;
        if (numClasses==1){
            defaultClass = mostCommonElement(n);
            totalBits = functionL(longitud,0,longitud);
        }else{
            double bits;
            int longitudCadena;
            for(int i = numClasses; i>1; i--){
                longitudCadena = n.size();
                String commonElement = mostCommonElement(n);
                if(i==numClasses){
                    defaultClass = commonElement;
                }
                int countElement = Collections.frequency(n, commonElement);
                bits = functionL(longitudCadena,longitudCadena-countElement,longitudCadena);
                totalBits += bits;
                n.removeAll(Collections.singletonList(commonElement));
            }
        }
        return new Pair<>(totalBits,defaultClass);
    }
    
    public static Double bitsExcepciones(ArrayList<String> n,int j,String firstClass){
        int longitud = n.size();
        HashSet<String> uniqueClass = new HashSet<>(n);
        int numClasses = uniqueClass.size();
        double totalBits = 0;
        if (numClasses==1){
            totalBits = functionL(longitud,0,longitud);
        }else{
            double bits;
            for(int i = numClasses; i>1; i--){
                String commonElement = mostCommonElement(n);
                if(i==numClasses){
                    commonElement = firstClass;
                }
                int countElement = Collections.frequency(n, commonElement);
                bits = functionL(longitud,longitud-countElement,longitud);
                totalBits += bits;
                n.removeAll(Collections.singletonList(commonElement));
            }
        }
        return totalBits;
    }
    
    public static double functionL(int n, int k, int b){
        BigInteger x = BigInteger.valueOf(b).add(new BigInteger("1"));
        BigInteger y = fact(n).divide(fact(k).multiply(fact(n-k)));
        double r = logb(x)/logb(new BigInteger("2")) + logb(y)/logb(new BigInteger("2"));
        return r;
    }
    
    public static BigInteger fact(int n){
        BigInteger factorial = BigInteger.valueOf(1);
        for (int i = n; i > 0; i--) {
            factorial = factorial.multiply(BigInteger.valueOf(i));
        }
        return factorial;
    }
    
    public static double logb(BigInteger val) {
        int n = val.bitLength();
        long mask = 1L << 52;
        long mantissa = 0;
        int j = 0;
        for (int i = 1; i < 54; i++) {
            j = n - i;
            if (j < 0) {
                break;
            }

            if (val.testBit(j)) {
                mantissa |= mask;
            }
            mask >>>= 1;
        }
        if (j > 0 && val.testBit(j - 1)) {
            mantissa++;
        }
        double f = mantissa / (double) (1L << 52);
        return (n - 1 + Math.log(f) * 1.44269504088896340735992468100189213742664595415298D);
    }
    
    private static String mostCommonElement(ArrayList<String> list) {
     
    Map<String, Integer> map = new HashMap<>();
     
    for(int i=0; i< list.size(); i++) {
         
        Integer frequency = map.get(list.get(i));
        if(frequency == null) {
            map.put(list.get(i), 1);
        } else {
            map.put(list.get(i), frequency+1);
        }
    }
     
    String mostCommonKey = null;
    int maxValue = -1;
    for(Map.Entry<String, Integer> entry: map.entrySet()) {
         
        if(entry.getValue() > maxValue) {
            mostCommonKey = entry.getKey();
            maxValue = entry.getValue();
        }
    }
     
    return mostCommonKey;
}
}
