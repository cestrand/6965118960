package dataset.diabetes;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.Data;
import weka.Matrix;
import weka.Wykres;
import weka.SiecB;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class TestyDiaNew {

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		Instances daneOrg = dane;

		daneOrg.setClassIndex(dane.numAttributes()-1);
		dane = Data.discretizeAttributesUnsupervised(dane, "3-4", 4, true);

		//klasyfikacja
		int par=2;
		SiecB bn = SiecB.createHillClimbGlobal(par); //parametry S� - domy�lne
		bn.buildClassifier(dane);
		bn.PrintScore();
		bn.PrintData();

		System.out.println("xWalidacja OneOut");
		int[][] M = bn.confMatrixOfXVal(dane, 0);  //na daneOrg - nie dzia�a
		Matrix.show(M);


		int q=2;
		System.out.println("xWalidacja "+q+"-krotna");
		double[] m01=new double[150];
		double[] m10=new double[150];
		int NN = 1000;  //aby szybko
		for(int i=1; i<NN; i++) { //246, 340, 581, 661 ,...
			M = bn.confMatrixOfXVal(dane, q, new Random(i));
			m01[M[0][1]]++;
			m10[M[1][0]]++;
		}
		for (int j=0; j<m01.length; j++)
			m01[j] = m01[j]/NN;
		for (int j=0; j<m10.length; j++)
			m10[j] = m10[j]/NN;
		Wykres w = new Wykres("Walidacja "+q+"-krotna - diabetes; "+NN+" pr�b, HCg"+par,true);
		double[][] M01 = new double[2][150]; 
		for(int i=0; i<150; i++) {
			M01[0][i] = i;
		}
		M01[1] = m01;
		double[][] M10 = new double[2][];
		M10[0] = M01[0];
		M10[1] = m10;
		w.pokaz("M01", M01);
		w.pokaz("M10", M10);
		//warto�� oczekiwana i odchylenie:
		double em01=0, em10=0;
		double st01=0, st10=0;
		for(int i=0; i<150; i++) {
			em01 = em01 + m01[i]*i;
			em10 = em10 + m10[i]*i;
		}
		for(int i=0; i<150; i++) {
			st01 = st01 + m01[i]*i*i;
			st10 = st10 + m10[i]*i*i;
		}
		st01 = Math.sqrt(st01-em01*em01);
		st10 = Math.sqrt(st10-em10*em10);
		System.out.printf("Oczekiwane b��dy: %3.1f %3.1f\n",em01,em10);
		System.out.printf("Odchylenia stand: %3.2f %3.2f\n",st01,st10);
		double uf01=0, uf10=0;
		
		
		System.out.println("Jedna z wart. skrajnych: ");
		M = bn.confMatrixOfXVal(dane, q, new Random(95924));  //wredne
		Matrix.show(M);
		
		NN = 10000;
		double[] org = dane.attributeToDoubleArray(dane.classIndex()); 
		int N = dane.numInstances();
		double[] licz = new double[N]; //licznik b��d�w
		for(int i=1; i<NN; i++) { 
			double[] dec = bn.xVal(dane, 2, new Random(i));
			for(int j=0; j<N; j++)
				if (dec[j]!=org[j]) licz[j]++;
		}
		double max=0;
		for(int j=0; j<N; j++) {
			if (max < licz[j])
				max = licz[j];
		}
		System.out.println("Max err = "+max);
		int cnt=0;  //.9 ->119, .99 -> 99 dla 10K, 98 dla 100K
		for(int j=0; j<N; j++) {
			if (max*.60 < licz[j]) {		//90% b��dnych decyzji - 119 inst
											//60% -- 151
				System.out.printf("%3d: %3d\n",j,(int)licz[j]);
				cnt++;
			}
		}
		System.out.println("Skrajnych: "+cnt);
		int[] dziw60 = new int[cnt];
		cnt = 0;
		for(int j=0; j<N; j++) {
			if (max*.60 < licz[j]) {		//60% b��dnych decyzji - 151 inst
				dziw60[cnt]= j;
				cnt++;		//nast�pne wolne miejsce
			}
		}
		//Podzia� danych na "zwykle" i "dziwne"
		Instances dziwne = new Instances(dane, cnt);    //nowe, puste
		for(int i=0; i<cnt; i++)
			dziwne.add(dane.get(dziw60[i]));
		Instances zwykle = new Instances(dane); 		//kopia, do usuni�cia
		for(int i=cnt-1; i>=0; i--)
			zwykle.remove(dziw60[i]);
		System.out.println("Po rozdzieleniu: "+zwykle.numInstances()+"+"+dziwne.numInstances());
		System.out.println("xWalidacja OneOut - dla zwyk�ych");
		M = bn.confMatrixOfXVal(zwykle, 0);  //na daneOrg - nie dzia�a
		Matrix.show(M);
		System.out.println(".. i dla dziwnych");
		M = bn.confMatrixOfXVal(dziwne, 0);  //na daneOrg - nie dzia�a
		Matrix.show(M);
		// NOWE ZADANIE: jak podzielić dane na NOWE 2 klasy: (zwykłe, dziwne)
		// Na ile stabila jest x-walidacja w każdym z tych zbiorów?
//................................................................................	
		Instances wzZw = new Instances(zwykle); //trzeba zmienić atrybut klasowy
		Instances wzDz = new Instances(dziwne);
		wzDz.setClassIndex(-1);
		wzZw.setClassIndex(-1);
		wzDz.deleteAttributeAt(dane.classIndex());
		wzZw.deleteAttributeAt(dane.classIndex());
		List<String> typyZw = new ArrayList<>(2); 
		typyZw.add("zw"); 
		typyZw.add("dz"); 
		Attribute att = new Attribute("Ordinary",typyZw);
		wzDz.insertAttributeAt(att, wzDz.numAttributes());
		for(int i=0; i<wzDz.size(); i++)
			wzDz.instance(i).setValue(wzDz.numAttributes()-1, "dz");
		wzZw.insertAttributeAt(att, wzZw.numAttributes());
		for(int i=0; i<wzZw.size(); i++)
			wzZw.instance(i).setValue(wzZw.numAttributes()-1, "zw");
		// po��czy� oba zbiory, ustawi� atryb. klasowy -> utworzy� klasyfikator
		
		System.out.println(wzDz.firstInstance());
		
		//		Arrays.sort(licz);
//		weka.Wykres wl = new weka.Wykres("��czna liczba b��d�w", false);
//		double[][] perr = new double[2][N];
//		for(int i=0; i<N; i++)
//			perr[0][i]=i;
//		perr[1] = licz;
//		wl.pokaz("NN="+NN, perr);
	}

}

//xWalidacja OneOut (z dodatk. dyskretyzacj� - 4 koszyki)
//416  84 
// 86 182 
//xWalidacja 2-krotna
//Oczekiwane b��dy: 82,0 85,5    - N=10000
//Odchylenia std.:   4,75 4,27
//Oczekiwane b��dy: 82,0 85,5    - N=100000
//Odchylenia std.:   4,66 4,18
//Odchylenia std.:   4,68 4,20


//	Tablica maksymalnej cz�sto�ci - z wykresu!: (bez 2 atrybut�w) 
//424  76
// 89 179  
//	xWalidacja OneOut (bez 2 atrybut�w) 
//418  82 
// 87 181 
//xWalidacja 2-krotna,  seed=123
//422  78 
// 87 181 
//xWalidacja 2-krotna,  seed=12
//427  73 
// 98 170 

//246: Err = 184
//340: delta = 34
//581: Err = 186
//661: delta = 42
//12071: Err = 186
//12252: Err = 187
//13185: delta = 40
//14410: delta = 41
//20013: Err = 187
//20025: delta = 34
//20282: delta = 37
//20289: delta = 40
//20409: delta = 41
//31456: Err = 188
//32417: delta = 40
//35578: delta = 41
//36254: Err = 190
//37442: delta = 42
//45606: delta = 43
//49131: Err = 189
//52242: Err = 192
//65000: delta = 45
	//xWalidacja 2-krotna
	//440  60 
	//105 163 
//67443: Err = 194
	//xWalidacja 2-krotna
	//408  92 
	//102 166
//95924: Err = 195
	//xWalidacja 2-krotna
	//403  97 
	// 98 170 

//Max err = 99999
//6: 99999
//9: 99998
//19: 99095
//36: 99984
//38: 99999
//41: 99999
//46: 99734
//54: 99999
//58: 99746
//70: 99999
//93: 99999
//95: 99998
//107: 99999
//109: 99999
//116: 99981
//124: 99993
//125: 99999
//138: 99751
//153: 99154
//170: 99984
//197: 99999
//198: 99855
//212: 99999
//213: 99654
//218: 99999
//219: 99980
//223: 99999
//228: 99999
//242: 99999
//244: 99998
//247: 99990
//248: 99997
//254: 99491
//260: 99999
//264: 99983
//267: 99104
//269: 99999
//276: 99989
//281: 99999
//282: 99999
//284: 99999
//285: 99989
//286: 99999
//291: 99999
//303: 99996
//308: 99997
//312: 99999
//321: 99999
//322: 99999
//327: 99999
//328: 99999
//335: 99993
//337: 99984
//349: 99999
//361: 99999
//364: 99997
//366: 99999
//400: 99999
//406: 99983
//419: 99999
//429: 99220
//436: 99999
//444: 99852
//448: 99999
//469: 99151
//470: 99138
//473: 99983
//486: 99162
//487: 99999
//489: 99996
//499: 99999
//502: 99999
//515: 99998
//549: 99999
//560: 99978
//568: 99997
//594: 99988
//619: 99992
//622: 99999
//638: 99567
//645: 99999
//657: 99992
//659: 99999
//664: 99279
//667: 99934
//670: 99999
//678: 99995
//701: 99999
//706: 99997
//709: 99999
//719: 99999
//730: 99814
//739: 99982
//744: 99999
//756: 99999
//757: 99978
//763: 99999
//766: 99978
//Skrajnych: 98