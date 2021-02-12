package projekt.tpt48;

import weka.Matrix;
import weka.SiecB;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.HashMap;

public class Mushrooms {
    public static void main(String[] args) throws Exception {
        Instances dane = ConverterUtils.DataSource.read("zasoby/mushrooms.arff");
//        wypiszWszystkieAtrybuty(dane);
        dane.setClass(dane.attribute("class"));
        System.out.println(dane);

        SiecB bn = SiecB.createK2(2);
        bn.buildClassifier(dane);
        System.out.println(bn);
/* Bayes Network Classifier
not using ADTree
#attributes=23 #classindex=0
Network structure (nodes followed by parents)
class(2):
cap-shape(6): class
cap-surface(4): class cap-shape
cap-color(10): class cap-surface
bruises(2): class cap-color
odor(9): class cap-color
gill-attachment(2): class cap-surface
gill-spacing(2): class bruises
gill-size(2): class cap-color
gill-color(12): class cap-color
stalk-shape(2): class gill-color
stalk-root(4): class odor
stalk-surface-above-ring(4): class bruises
stalk-surface-below-ring(4): class cap-color
stalk-color-above-ring(9): class cap-color
stalk-color-below-ring(9): class stalk-color-above-ring
veil-type(1): class
veil-color(4): class gill-attachment
ring-number(3): class gill-color
ring-type(5): class gill-color
spore-print-color(9): class gill-color
population(6): class ring-type
habitat(7): class population
LogScore Bayes: -113231.81996814942
LogScore BDeu: -120178.13006386424
LogScore MDL: -120481.98150720591
LogScore ENTROPY: -113842.58028921449
LogScore AIC: -115317.58028921449 */

    Matrix.show(bn.confMatrix(dane));
/*
3890   26
4    4204
*/

        HashMap<String, Integer> gill_attachment_occurences = policzWystapieniaAtrybutu(dane, 6);
        System.out.println(gill_attachment_occurences);
        // {a=210, f=7914}

        Instances dane_bez_gill_attachment = new Instances(dane);
        dane_bez_gill_attachment.deleteAttributeAt(6);
        SiecB bn2 = SiecB.createK2(2);
        bn2.buildClassifier(dane_bez_gill_attachment);
        System.out.println(bn2);
        Matrix.show(bn2.confMatrix(dane_bez_gill_attachment));
/*
  3890  26
  4 4204
*/

        HashMap<String, Integer> bruises_occurences = policzWystapieniaAtrybutu(dane, 4);
        System.out.println(bruises_occurences);
        // {t=3376, f=4748}

        weka.attributeSelection.CorrelationAttributeEval corrEval = new weka.attributeSelection.CorrelationAttributeEval();
        corrEval.setOutputDetailedInfo(true);
        corrEval.buildEvaluator(dane);
        System.out.println(corrEval);
/*  Correlation Ranking Filter
	Detailed output for nominal attributes

cap-shape
	x: 0.026886
	b: 0.182567
	s: 0.060664
	f: 0.018526
	k: 0.163565
	c: 0.023007

cap-surface
	s: 0.095454
	y: 0.088677
	f: 0.195415
	g: 0.023007

cap-color
	n: 0.04436
	y: 0.113014
	w: 0.133683
	g: 0.046456
	e: 0.097112
	p: 0.034702
	b: 0.067544
	u: 0.042854
	c: 0.03091
	r: 0.042854

bruises
	t: 0.50153
	f: 0.50153

odor
	p: 0.186984
	a: 0.219529
	l: 0.219529
	n: 0.785557
	f: 0.623842
	c: 0.161278
	y: 0.28636
	s: 0.28636
	m: 0.069159

gill-attachment
	f: 0.1292
	a: 0.1292

gill-spacing
	c: 0.348387
	w: 0.348387

gill-size
	n: 0.540024
	b: 0.540024

gill-color
	k: 0.149641
	n: 0.288943
	g: 0.120285
	p: 0.05038
	w: 0.231316
	h: 0.150694
	u: 0.195359
	e: 0.105491
	b: 0.538808
	r: 0.056426
	y: 0.046828
	o: 0.085962

stalk-shape
	e: 0.102019
	t: 0.102019

stalk-root
	e: 0.202839
	c: 0.218548
	b: 0.351508
	r: 0.150087

stalk-surface-above-ring
	s: 0.491314
	f: 0.119503
	k: 0.587658
	y: 0.016198

stalk-surface-below-ring
	s: 0.425444
	f: 0.136782
	y: 0.081674
	k: 0.573524

stalk-color-above-ring
	w: 0.21774
	g: 0.266489
	p: 0.230277
	n: 0.233164
	b: 0.245662
	e: 0.105491
	o: 0.150087
	c: 0.069159
	y: 0.032545

stalk-color-below-ring
	w: 0.214112
	p: 0.230277
	g: 0.266489
	b: 0.245662
	n: 0.203966
	e: 0.105491
	y: 0.056426
	o: 0.150087
	c: 0.069159

veil-type
	p: 0

veil-color
	w: 0.140541
	n: 0.105491
	o: 0.105491
	y: 0.032545

ring-number
	o: 0.182101
	t: 0.2046
	n: 0.069159

ring-type
	p: 0.540469
	e: 0.223286
	l: 0.451619
	f: 0.074371
	n: 0.069159

spore-print-color
	k: 0.396832
	n: 0.416645
	u: 0.074371
	h: 0.490229
	w: 0.357384
	r: 0.098024
	o: 0.074371
	y: 0.074371
	b: 0.074371

population
	s: 0.159572
	n: 0.219529
	a: 0.214871
	v: 0.443722
	y: 0.107055
	c: 0.137645

habitat
	u: 0.112078
	g: 0.165004
	m: 0.138627
	d: 0.126123
	p: 0.323346
	w: 0.150087
	l: 0.15515

         */

        Instances dane2 = new Instances(dane_bez_gill_attachment);
        dane2.deleteAttributeAt(dane2.attribute("veil-type").index());
        dane2.deleteAttributeAt(dane2.attribute("stalk-shape").index());

        SiecB bn3 = SiecB.createK2(2);
        bn3.buildClassifier(dane2);
        System.out.println(bn3);

/* Bayes Network Classifier
not using ADTree
#attributes=20 #classindex=0
Network structure (nodes followed by parents)
class(2):
cap-shape(6): class
cap-surface(4): class cap-shape
cap-color(10): class cap-surface
bruises(2): class cap-color
odor(9): class cap-color
gill-spacing(2): class bruises
gill-size(2): class cap-color
gill-color(12): class cap-color
stalk-root(4): class odor
stalk-surface-above-ring(4): class bruises
stalk-surface-below-ring(4): class cap-color
stalk-color-above-ring(9): class cap-color
stalk-color-below-ring(9): class stalk-color-above-ring
veil-color(4): class stalk-color-above-ring
ring-number(3): class gill-color
ring-type(5): class gill-color
spore-print-color(9): class gill-color
population(6): class ring-type
habitat(7): class population
LogScore Bayes: -109820.42273982955
LogScore BDeu: -116851.28336818366
LogScore MDL: -117167.16677404134
LogScore ENTROPY: -110482.75266643638
LogScore AIC: -111967.75266643638  */

    }

    private static HashMap<String, Integer> policzWystapieniaAtrybutu(Instances dane, int nrAtrybutu) {
        HashMap<String, Integer> gill_attachment_occurences = new HashMap<String, Integer>();
        for(int i = 0; i< dane.numInstances(); i++) {
            int ilosc_wystapien = gill_attachment_occurences.getOrDefault(dane.get(i).stringValue(nrAtrybutu), 0);
            gill_attachment_occurences.put(dane.get(i).stringValue(nrAtrybutu), ilosc_wystapien+1);
        }
        return gill_attachment_occurences;
    }

    private static void wypiszWszystkieAtrybuty(Instances dane) {
        for (Attribute attr : java.util.Collections.list(dane.enumerateAttributes())) {
            System.out.println(attr);
        }
    }
}
