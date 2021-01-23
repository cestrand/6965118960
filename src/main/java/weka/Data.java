package weka;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Data {
    /**
     * Dyskretyzacja polega na zamianie atrybutów numerycznych na nominalne.
     * @param dane Jakie dane mają być poddane dyskretyzacji
     * @param kolumny a string representing the list of attributes. attributes are indexed from 1.
     *          eg: first-3,5,6-last
     * @param bins Na ile przedziałów należy podzielić
     * @param equalFrequency Jeśli true to dzieli na przedziały o równej liczności,
     *                         jeśli false to dzieli na przedziały o równej szerokości.
     * @return Zdyskretyzowany zbiór danych
     * @throws Exception
     */
    public static Instances discretizeAttributesUnsupervised(Instances dane, String kolumny, int bins, boolean equalFrequency) throws Exception {
        weka.filters.unsupervised.attribute.Discretize dis = new weka.filters.unsupervised.attribute.Discretize();
        dis.setAttributeIndices(kolumny);
        dis.setBins(bins);
        dis.setUseEqualFrequency(equalFrequency);
        dis.setInputFormat(dane);
        return Filter.useFilter(dane, dis);
    }

    /**
     * Dyskretyzacja polega na zamianie atrybutów numerycznych na nominalne.
     * @param dane Jakie dane mają być poddane dyskretyzacji
     * @param kolumny a string representing the list of attributes. attributes are indexed from 1.
     *          eg: first-3,5,6-last
     * @return Zdyskretyzowany zbiór danych
     * @throws Exception
     */
    public static Instances discretizeAttributesSupervised(Instances dane, String kolumny) throws Exception {
        weka.filters.supervised.attribute.Discretize dis = new weka.filters.supervised.attribute.Discretize();
        dis.setAttributeIndices(kolumny);
        dis.setInputFormat(dane);
        return Filter.useFilter(dane, dis);
    }

    /**
     * Dzieli zbiór danych na te które zawierają wiersze o numerach zawartych w numeryWierszy i pozostałe.
     * @param numeryWierszy Lista wierszy które chcemy wydobyć.
     * @param dane
     * @return Obiekt klasy Podzial; gdzie w Podzial.l znajdują się te wydobyte, zaś w Podzial.r pozostałe.
     */
    public static Podzial Split(ArrayList<Integer> numeryWierszy, Instances dane) {

        ArrayList<Integer> posortowane = new ArrayList<Integer>(numeryWierszy);
        Collections.sort(posortowane);
        Collections.reverse(posortowane);
        Instances inne = new Instances(dane);
        for (int i : posortowane) {
            inne.add(dane.get(i));
        }

        Instances dobre = new Instances(dane);
        for(int i: posortowane) {
            dobre.remove(i);
        }
        return new Podzial(inne, dobre);
    }

    /**
     * Lączy ze sobą dane z podziału spowrotem w jeden zbiór danych dodając atrybut wskazujący na przynależność
     * do wskazanego podzbioru.
     * @param p Podział danych
     * @param nazwaKolumny Nazwa atrybutu który zostanie dodany
     * @param lNazwa Wartość atrybutu która zostanie nadana wierszom ze zbioru p.l
     * @param rNazwa Wartość atrybutu któa zostanie nadana wierszom ze zbioru p.r
     * @return
     */
    public static Instances DodajKolumneWzgledemPodzialu(Podzial p, String nazwaKolumny, String lNazwa, String rNazwa) {
        List<String> typyZw = new ArrayList<>(2);
        typyZw.add(lNazwa);
        typyZw.add(rNazwa);
        Attribute att = new Attribute(nazwaKolumny, typyZw);
        Instances lDane = new Instances(p.l);
        Instances rDane = new Instances(p.r);

        lDane.insertAttributeAt(att, lDane.numAttributes());
        for (int i = 0; i < lDane.size(); i++) {
            lDane.instance(i).setValue(lDane.numAttributes() - 1, lNazwa);
        }
        rDane.insertAttributeAt(att, rDane.numAttributes());
        for (int i = 0; i < rDane.size(); i++) {
            rDane.instance(i).setValue(rDane.numAttributes() - 1, rNazwa);
        }

        Instances polaczoneDane = new Instances(rDane, rDane.numInstances()+lDane.numInstances());
        polaczoneDane.addAll(lDane);
        return polaczoneDane;
    }

}

