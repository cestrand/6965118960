package projekt.janepe;

import org.junit.jupiter.api.Test;
import weka.Matrix;
import weka.SiecB;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class SieciBayes {
    private void evaluateModel(Instances dane, SiecB bn) throws Exception {
        Evaluation ev = new Evaluation(dane);
        ev.evaluateModel(bn, dane);
        System.out.printf("Poprawne:\t%d --- %.2f%%\n", (int) ev.correct(), ev.correct() / ev.numInstances() * 100);
        System.out.printf("Błędne:  \t%d --- %.2f%%\n", (int) ev.incorrect(), ev.incorrect() / ev.numInstances() * 100);
    }

    @Test
    public void SiecBK2P3() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));
        SiecB bn = SiecB.createK2(3);
        bn.buildClassifier(dane);
        Matrix.show(bn.confMatrix(dane));

        evaluateModel(dane, bn);
    }

    @Test
    public void SiecBHCGP3() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));
        SiecB bn = SiecB.createHillClimbGlobal(3);
        bn.buildClassifier(dane);
        Matrix.show(bn.confMatrix(dane));

        evaluateModel(dane, bn);
    }

    @Test
    public void SiecBHCLP3() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));
        SiecB bn = SiecB.createHillClimbLocal(3);
        bn.buildClassifier(dane);
        Matrix.show(bn.confMatrix(dane));

        evaluateModel(dane, bn);
    }
}
