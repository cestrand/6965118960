package dataset.weather;

import weka.Matrix;
import weka.SiecB;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.estimators.Estimator;

public class Testy {

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/weather.nominal.arff");
		dane.setClassIndex(4);

		//klasyfikacja
		SiecB bn = SiecB.createHillClimbGlobal(2); //parametry S� - domy�lne
		bn.buildClassifier(dane);
		bn.PrintScore();

		wypiszObliczonePrawdopodobienstwaIBledyKlasyfikacji(dane, bn);

		Estimator[][] cpt = bn.m_Distributions;
		System.out.println("Estymator dla przypadku: nr 0=Outlook, 0=yes(play)");
		System.out.println(cpt[0][0]);  //0=Outlook, 0=yes (dla play)
//		double[] d = bn.distributionForInstance(wiersz);
//		System.out.printf("%4.2f; %4.2f \n",d[0],d[1]);



		System.out.println("xWalidacja OneOut"); // TO JUŻ MAMY GDZIES ZROBIONE
		for(int i = 0; i< dane.numInstances(); i++) {
			Instance wiersz = dane.get(i);
			// usun�� z danych i przeliczy� CPTs
			bn.initCPTs(); 				//tworzy i inicjuje NOW� tablic� CPT
			wiersz.setWeight(0);    	//wy��cza obs�ug�, pomija
			for(int j = 0; j< dane.numInstances(); j++) {
			   bn.updateClassifier(dane.get(j));  //pomijamy i-ty wiersz
			}
			wiersz.setWeight(1);		//wiersz wraca do danych
			int dec = (int) bn.classifyInstance(wiersz); //nowa decyzja
//			System.out.print(i+": "+wiersz);
			char c;
			if (dec==wiersz.classValue())
				c=' ';
			else c='*';
			double[] prob = bn.distributionForInstance(wiersz); //nowy wynik
			System.out.printf("%2d: [%4.2f; %4.2f] %c\n",i,prob[0],prob[1],c);
		}

		bn.estimateCPTs(dane); //odtwarza tablic� CPT


		double[] dec = bn.xOneOut(dane); // OOOO :3 TUUU
		
		int[][] M = new int[dane.numClasses()][dane.numClasses()];
		int errs = 0;  	//por�wnanie - krok po kroku...
		for(int i = 0; i< dane.numInstances(); i++) {
			Instance wiersz = dane.get(i);
//			M[(int)wiersz.classValue()][(int)dec[i]]++; //macierz b��d�w
			if (dec[i] != wiersz.classValue()) errs++;
		}
		double[] org = dane.attributeToDoubleArray(dane.classIndex());
		System.out.println("Liczba b��d�w: "+errs);
		     			// ... i ca�o�ciowe
		M = Matrix.confMatrix(org, dec, bn.m_Instances.numClasses());
		Matrix.show(M);
		
		//bn.condProb(W=True|O=sunny,P=yes)
		//bn.condProb(W;sunny,?,?,True,yes)
//		double p = bn.condProb(3,0.,-1.,-1.,0.,0.);
//		System.out.println("Prob(W=True|O=sunny,P=yes)="+p);
		
		//bn.condProb(H=high|O=rainy,P=yes)
		//bn.condProb(H;rainy,?,high,?,yes)
//		p=bn.condProb(2,2.,-1.,0.,-1.,0.);
//		System.out.println("Prob(H=high|O=rainy,P=yes)="+p);
		
		//bn.condProb(H=high|O=rainy)=? bn.condProb(H=normal|O=rainy)=?
		// Prob(H=high|O=rainy)=Prob(H=high,O=rainy)/Prob(O=rainy)=....
		double pr = bn.prob(2,-1,0,-1,-1)/bn.prob(2,-1,-1,-1,-1);
//		p=bn.condProb(2,0.,-1.,1.,-1.,1.);
		System.out.println("Prob(H=high|O=rainy)="+pr);
		pr = bn.prob(2,-1,1,-1,-1)/bn.prob(2,-1,-1,-1,-1);
		System.out.println("Prob(H=normal|O=rainy)="+pr);
		System.out.println("Suma kontrolna="+bn.prob(-1,-1,-1,-1,-1));

//............... Zapis do pliku jako bn.saveToBIFF(nazwaPliku)
//		String biff = bn.toXMLBIF03();  //weatherHCg2
//        FileWriter fw = null;
////        if (!nazwaPliku.toLowerCase().endsWith(".xml"))
////           nazwaPliku+= ".xml";
//        fw = new FileWriter("weatherHCg2.xml");
//        // ca�� tablic� liczby[]
//        fw.write(biff);
//        fw.close();
		
	}

	private static void wypiszObliczonePrawdopodobienstwaIBledyKlasyfikacji(Instances dane, SiecB bn) throws Exception {
		for(int i = 0; i< dane.numInstances(); i++) {
			Instance wiersz = dane.get(i);
			int dec = (int) bn.classifyInstance(wiersz);
			char c;
			if (dec==wiersz.classValue())
				c=' ';
			else c='*';
			double[] prob = bn.distributionForInstance(wiersz);
			System.out.printf("%2d: [%4.2f; %4.2f] %c\n",i,prob[0],prob[1],c);
		}
	}

}

//Sprawdzić:
//Scheme:       weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.global.HillClimber -- -R -P 2 -S LOO-CV -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5


//Network structure (nodes followed by parents) HClocal -2
//outlook(3): play 
//temperature(3): humidity 
//humidity(2): play 
//windy(2): play outlook 
//play(2): 
//LogScore Bayes: -65.33162426161034
//LogScore BDeu: -90.14814569508599
//LogScore MDL: -86.872164964719
//LogScore ENTROPY: -64.4401776629893
//LogScore AIC: -81.4401776629893
//   HC local           HC global           K2 local
// 0: [0,22; 0,78]    0: [0,10; 0,90]     0: [0,09; 0,91] 
// 1: [0,32; 0,68]    1: [0,16; 0,84]     1: [0,14; 0,86] 
// 2: [0,82; 0,18]    2: [0,91; 0,09]     2: [0,89; 0,11] 
// 3: [0,79; 0,21]    3: [0,85; 0,15]     3: [0,86; 0,14] 
// 4: [0,95; 0,05]    4: [0,91; 0,09]     4: [0,88; 0,12] 
// 5: [0,37; 0,63]    5: [0,22; 0,78]     5: [0,17; 0,83] 
// 6: [0,96; 0,04]    6: [0,91; 0,09]     6: [0,90; 0,10] 
// 7: [0,22; 0,78]    7: [0,10; 0,90]     7: [0,32; 0,68] 
// 8: [0,61; 0,39]    8: [0,80; 0,20]     8: [0,73; 0,27] 
// 9: [0,95; 0,05]    9: [0,91; 0,09]     9: [0,97; 0,03] 
//10: [0,73; 0,27]   10: [0,87; 0,13]    10: [0,80; 0,20]
//11: [0,82; 0,18]   11: [0,91; 0,09]    11: [0,83; 0,17]
//12: [0,96; 0,04]   12: [0,91; 0,09]    12: [0,98; 0,02]
//13: [0,09; 0,91]   13: [0,14; 0,86]    13: [0,15; 0,85]
                                               