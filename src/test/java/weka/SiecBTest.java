package weka;
import org.junit.jupiter.api.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

public class SiecBTest {
    @Test
    void createK2SearchAlgorithmType() {
        SiecB s = SiecB.createK2(5);
        assertTrue(s.getSearchAlgorithm() instanceof weka.classifiers.bayes.net.search.local.K2);
    }

    @Test
    void loadingBIFFile() throws Exception {
        SiecB s = SiecB.loadFromBIFAndNormalize("zasoby/DiaInne.xml");
    }

    @Test
    void loadDiabetes() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/weka-data/diabetes.arff");
        assertEquals(dane.size(), 768);
    }

    @Test
    void settingClassAttribute() throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/weka-data/diabetes.arff");
        dane.setClassIndex(dane.numAttributes() - 1);

        Instances dane2 = ConverterUtils.DataSource.read("zasoby/weka-data/diabetes.arff");
        dane2.setClass(dane2.attribute("class"));

        assertEquals(dane.classIndex(), dane2.classIndex());
    }


    @Test
    void discretizeFunnyTest() throws Exception {
        Discretize dis = new Discretize();

        // zróbmy sobie dane dla których przetestujemy dyskretyzator
        Attribute attr1 = new Attribute("atrybut");
        ArrayList<Attribute> atrybuty = new ArrayList<Attribute>();
        atrybuty.add(attr1);
        Instances dane = new Instances("testData", atrybuty, 0);
        dane.add(new DenseInstance(1, new double[]{0.3}));
        dane.add(new DenseInstance(1, new double[]{0.5}));
        dane.add(new DenseInstance(1, new double[]{0.7}));
        dane.add(new DenseInstance(1, new double[]{0.9}));
        dane.add(new DenseInstance(1, new double[]{0.55}));
        assertEquals(dane.toString(), "@relation testData\n" +
                "\n" +
                "@attribute atrybut numeric\n" +
                "\n" +
                "@data\n" +
                "0.3\n" +
                "0.5\n" +
                "0.7\n" +
                "0.9\n" +
                "0.55");
        dis.setBins(2);
        dis.setInputFormat(dane);
        Instances zdyskretyzowane = Filter.useFilter(dane, dis);
        assertEquals(zdyskretyzowane.toString(), "@relation testData-weka.filters.unsupervised.attribute.Discretize-B2-M-1.0-Rfirst-last-precision6\n" +
                "\n" +
                "@attribute atrybut {'\\'(-inf-0.6]\\'','\\'(0.6-inf)\\''}\n" +
                "\n" +
                "@data\n" +
                "'\\'(-inf-0.6]\\''\n" +
                "'\\'(-inf-0.6]\\''\n" +
                "'\\'(0.6-inf)\\''\n" +
                "'\\'(0.6-inf)\\''\n" +
                "'\\'(-inf-0.6]\\''");
    }




}