/*
W tym pliku znajdują się testy z użyciem JUnit 5.
Poszczególne testy można łatwo uruchamiać niezależnie w IDE (ja używam IntelliJ IDEA).
*/

package projekt.janepe;

import org.junit.jupiter.api.Test;
import svm.SVMModel;
import svm.SVMNode;
import svm.SVMParameter;
import svm.SVMProblem;
import weka.Data;
import weka.Matrix;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
import weka.core.converters.LibSVMLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

public class SVM {

    @Test
    public void SVM1() throws Exception {
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
    public void SVMRBF1() throws Exception {
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
    public void SVMRBF2() throws Exception {
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

    @Test
    public void SVMLabT1() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/ph-data.arff");
        dane.setClass(dane.attribute("label"));

        Standardize flt = new Standardize();
        flt.setInputFormat(dane);
        dane = Filter.useFilter(dane, flt);

        SVMProblem problem = SVMProblem.fromInstances(dane);
        problem.par.svm_type = SVMParameter.C_SVC;
        problem.par.kernel_type = SVMParameter.RBF;
        problem.par.C = 100000;
        problem.par.gamma = 0.5;
        SVMModel model= problem.train();

        int[][] M = problem.confMatrix(model);
        Matrix.show(M);
        // policzymy sobie dokładność z użyciem kodu pani Aleksandry.
        projekt.essaerae.AnalizaGlass.pokazDokladnosc(dane, M);


        // Zrobimy jeszcze kroswalidacje, żeby pokazać, że umiemy robić takie fikołki.

        // == 4 krotna kroswalidacja ==
        int numFolds = 4;
        Random random = new Random(83838383);
        int poprawne = 0;
        int bledne = 0;
        int m = dane.numClasses();
        M = new int[m][m];

        double predictions[] = new double[dane.numInstances()];
        // Do the folds
        for (int i = 0; i < numFolds; i++) {
            Instances train = dane.trainCV(numFolds, i, random);
//            setPriors(train);
            SVMProblem pr = SVMProblem.fromInstances(train);
            pr.par = problem.par;
            SVMModel mo = pr.train();
            Instances test = dane.testCV(numFolds, i);
            SVMProblem prtest = SVMProblem.fromInstances(test);
            for(int j=0; j<prtest.l; j++) {
                double pred = mo.predict((SVMNode[]) prtest.x[j]);
                int i1 = (int) prtest.y[j];
                int i2 = (int) pred;
                M[i1][i2]++; //macierz błędów
                if(i1 == i2) {
                    poprawne++;
                }
                else {
                    bledne++;
                }
            }
        }
        Matrix.show(M);
        System.out.printf("Poprawne:\t%d --- %.2f%%\n", (int) poprawne, (double)poprawne / dane.numInstances() * 100);
        System.out.printf("Błędne:  \t%d --- %.2f%%\n", (int) bledne, (double)bledne / dane.numInstances() * 100);
        // no i otrzymaliśmy accuracy = 71,21% przy 4 krotnej kroswalidacji

        // albo jeszcze inaczej kroswalidacja z użyciem kodu napisanego na laboratorium
        double[] celX = new double[dane.numInstances()];
        double dokladnosc = problem.crossValidation(problem.par, 4, celX);
        System.out.printf("Dokładność:\t%.2f\n", dokladnosc);
    }

}
