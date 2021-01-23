package weka;

import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

/**
 * Klasa której celem jest przeprowadzenie wielokrotnej x-walidacji i przechowywanie jej wyniku.
 */
public class MultipleXVal {
    public int numberOfIterations;
    public int q;
    public Instances dane;
    public SiecB bn;
    public double[] bledyPerWiersz;

    public MultipleXVal(int numberOfIterations, int q, Instances dane, SiecB bn) {
        this.numberOfIterations = numberOfIterations;
        this.q = q;
        this.dane = dane;
        this.bn = bn;
    }

    public void Run() throws Exception {
        bledyPerWiersz = new double[dane.numInstances()];
        double[] org = dane.attributeToDoubleArray(dane.classIndex());
        for (int i = 1; i < numberOfIterations; i++) {
            double[] predykcja = bn.xVal(dane, 2, new Random(i));
            for (int j = 0; j < dane.numInstances(); j++)
                if (predykcja[j] != org[j])
                    bledyPerWiersz[j]++;
        }
    }

    public int SumOfErrors() {
        int suma = 0;
        for(int i = 0; i<bledyPerWiersz.length; i++) {
            suma += bledyPerWiersz[i];
        }
        return suma;
    }

    /**
     * Zwraca listę numerów tych wierszy, których błędy w x-walidacji wyniosły ponad pewną wartość.
     * @param errors Ostro powyżej tej liczby błędów dodaje ten wiersz do listy.
     * @return
     */
    public ArrayList<Integer> InstancesWithErrorsOver(int errors) {
        assert bledyPerWiersz.length > 0: "Najpierw wywołaj Run";
        ArrayList<Integer> l = new ArrayList<Integer>();
        for(int i = 0; i<bledyPerWiersz.length; i++) {
            if(bledyPerWiersz[i] > errors) {
                l.add(i);
            }
        }
        return l;
    }
}
