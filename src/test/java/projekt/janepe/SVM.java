/*
W tym pliku znajdują się testy z użyciem JUnit 5.
Poszczególne testy można łatwo uruchamiać niezależnie w IDE (ja używam IntelliJ IDEA).
*/

package projekt.janepe;

import org.junit.jupiter.api.Test;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
import weka.core.converters.LibSVMLoader;

import java.util.Random;

public class SVM {

    @Test
    public void SVMTest1() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));

        SMO smo = new SMO();
        // SMO is widely used for training support vector machines and is implemented by the popular LIBSVM tool.
        // The publication of the SMO algorithm in 1998 has generated a lot of excitement in the SVM community,
        // as previously available methods for SVM training were much more complex and required expensive third-party
        // QP solvers.[4]
        //
        // Multi-class problems are solved using pairwise classification (aka 1-vs-1).
        //  It also normalizes all attributes by default.
        // Domyślny kernel: (default: weka.classifiers.functions.supportVector.PolyKernel)

        // Lepszy wynik otrzymujemy przy tym problemie używając standaryzacji niż normalizacji.
        smo.setFilterType(new SelectedTag(smo.FILTER_STANDARDIZE, smo.TAGS_FILTER));
        smo.buildClassifier(dane);
        SieciBayes.evaluateModel(dane, smo);
    }

    @Test
    public void SVMRBFTest() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));

        SMO smo = new SMO();
        smo.setFilterType(new SelectedTag(smo.FILTER_STANDARDIZE, smo.TAGS_FILTER));
        RBFKernel kernel = new RBFKernel();
        smo.setKernel(kernel);
        smo.buildClassifier(dane);
        SieciBayes.evaluateModel(dane, smo);
    }

    @Test
    public void SVMRBFTest2() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));

        SMO smo = new SMO();
        smo.setFilterType(new SelectedTag(smo.FILTER_STANDARDIZE, smo.TAGS_FILTER));
        RBFKernel kernel = new RBFKernel();
        kernel.setGamma(0.5);
        smo.setKernel(kernel);
        smo.setC(100000);
        smo.buildClassifier(dane);
        SieciBayes.evaluateModel(dane, smo);

        Evaluation ev = new Evaluation(dane);
        StringBuffer xValOut = new StringBuffer();
        ev.crossValidateModel(smo, dane, 4, new Random(69), xValOut);
        System.out.println(xValOut);
        SieciBayes.printEvaluateModel(ev);
    }

}
