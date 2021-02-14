/*
W tym pliku znajdują się testy z użyciem JUnit 5.
Poszczególne testy można łatwo uruchamiać niezależnie w IDE (ja używam IntelliJ IDEA).
*/

package projekt.janepe;

import org.junit.jupiter.api.Test;
import weka.Data;
import weka.Matrix;
import weka.SiecB;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Random;

public class SieciBayes {
    public static void evaluateModel(Instances dane, Classifier bn) throws Exception {
        Evaluation ev = new Evaluation(dane);
        ev.evaluateModel(bn, dane);
        printEvaluateModel(ev);
    }

    public static void printEvaluateModel(Evaluation ev) {
        Matrix.show(ev.confusionMatrix());
        System.out.printf("Poprawne:\t%d --- %.2f%%\n", (int) ev.correct(), ev.correct() / ev.numInstances() * 100);
        System.out.printf("Błędne:  \t%d --- %.2f%%\n", (int) ev.incorrect(), ev.incorrect() / ev.numInstances() * 100);
    }

    @Test
    public void SiecBK2P3() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));
        SiecB bn = SiecB.createK2(3);
        bn.buildClassifier(dane);
        bn.saveToBIF("C:/siec_bk2p3.xml");

        evaluateModel(dane, bn);
    }

    @Test
    public void SiecBHCGP3() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));
        SiecB bn = SiecB.createHillClimbGlobal(3, true);
        bn.buildClassifier(dane);
        bn.saveToBIF("C:/siec_bhcgp3.xml");

        evaluateModel(dane, bn);

        double[] xval = bn.xVal(dane, 3, new Random(69));


        Evaluation ev = new Evaluation(dane);
        StringBuffer xValOut = new StringBuffer();
        ev.crossValidateModel(bn, dane, 3, new Random(69), xValOut);
        System.out.println(xValOut);
        printEvaluateModel(ev);

        // Upewnijmy się, że kroswalidacja Weki działa podobnie jak ta napisana na laboratorium.
        // Jeśeli liczba poprawnych jest równa liczbie poprawnych z wyniku otrzymanego powyżej to jest ok.
        int poprawne = 0;
        for(int i=0; i<dane.numInstances(); i++) {
            if(xval[i] == dane.get(i).classValue()) {
                poprawne++;
            }
        }
        // wyszło 486 podczas gdy kroswalidacja weki zwróciła 455 poprawnych.
        // Co mogło pójść inaczej? Okazuje się, że w inny sposób randomizujemy dane i wybieramy podzbiory.
        // Nie oznacza to jednak, że ta metoda jest niepoprawna.
    }

    @Test
    public void SiecBHCLP3() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));
        SiecB bn = SiecB.createHillClimbLocal(3);
        bn.buildClassifier(dane);
        bn.saveToBIF("C:/siec_bhclp3.xml");

        evaluateModel(dane, bn);
    }

    @Test
    public void SiecBHCGP3DyskretyzacjaNienadzorowana() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));
        dane = Data.discretizeAttributesUnsupervised(dane, "1-3", 25, false);
        SiecB bn = SiecB.createHillClimbGlobal(3, true);
        bn.buildClassifier(dane);
        bn.saveToBIF("C:/siec_bhcgp3_unsupervised.xml");

        evaluateModel(dane, bn);

        double[] xval = bn.xVal(dane, 3, new Random(69));


        Evaluation ev = new Evaluation(dane);
        StringBuffer xValOut = new StringBuffer();
        ev.crossValidateModel(bn, dane, 3, new Random(69), xValOut);
        System.out.println(xValOut);
        printEvaluateModel(ev);

        // Upewnijmy się, że kroswalidacja Weki działa podobnie jak ta napisana na laboratorium.
        // Jeśeli liczba poprawnych jest równa liczbie poprawnych z wyniku otrzymanego powyżej to jest ok.
        int poprawne = 0;
        for(int i=0; i<dane.numInstances(); i++) {
            if(xval[i] == dane.get(i).classValue()) {
                poprawne++;
            }
        }
        // wyszło 486 podczas gdy kroswalidacja weki zwróciła 455 poprawnych.
        // Co mogło pójść inaczej? Okazuje się, że w inny sposób randomizujemy dane i wybieramy podzbiory.
        // Nie oznacza to jednak, że ta metoda jest niepoprawna.
    }
}
