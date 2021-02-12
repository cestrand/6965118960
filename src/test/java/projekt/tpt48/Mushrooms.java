package projekt.tpt48;

import weka.Matrix;
import weka.SiecB;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.HashMap;
import java.util.Random;

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


        MultilayerPerceptron mp = new MultilayerPerceptron();
        mp.setHiddenLayers("5,3");
        mp.setLearningRate(0.3);
        //mp.setGUI(true);
        mp.buildClassifier(dane);
        System.out.println(mp);
//        Evaluation ev = new Evaluation(dane);
//        ev.crossValidateModel(mp, dane, 30, new Random(123));
//        System.out.println(ev.toMatrixString());
        /*
        Sigmoid Node 0
    Inputs    Weights
    Threshold    6.951822048272696
    Node 7    -5.13439415580438
    Node 8    -5.1536505961257815
    Node 9    -4.881627836676822
Sigmoid Node 1
    Inputs    Weights
    Threshold    -6.95181366558074
    Node 7    5.137563385464563
    Node 8    5.154763387872527
    Node 9    4.877300368476511
Sigmoid Node 2
    Inputs    Weights
    Threshold    -0.003156998446017421
    Attrib cap-shape=x    -0.050827544272416716
    Attrib cap-shape=b    0.5278329560694567
    Attrib cap-shape=s    -0.18191194131691257
    Attrib cap-shape=f    -0.212078075547399
    Attrib cap-shape=k    -0.07132452353394679
    Attrib cap-shape=c    0.17465007391917395
    Attrib cap-surface=s    0.05071546540241768
    Attrib cap-surface=y    0.06382327319213632
    Attrib cap-surface=f    -0.3996153210765516
    Attrib cap-surface=g    0.39624930277578924
    Attrib cap-color=n    -0.22834442683874204
    Attrib cap-color=y    -0.29529453628997665
    Attrib cap-color=w    0.36623015313638024
    Attrib cap-color=g    -0.05376818663351509
    Attrib cap-color=e    -0.05905454793921896
    Attrib cap-color=p    0.5407375710365963
    Attrib cap-color=b    0.29570253816352526
    Attrib cap-color=u    -0.006219814116874554
    Attrib cap-color=c    -0.22443956538811496
    Attrib cap-color=r    -0.1592269862506091
    Attrib bruises=f    -0.2380496845785668
    Attrib odor=p    0.9853590635684237
    Attrib odor=a    -0.9164509711974127
    Attrib odor=l    -1.1651212640325104
    Attrib odor=n    -1.7088288229359676
    Attrib odor=f    1.069619476222358
    Attrib odor=c    1.0246264666133018
    Attrib odor=y    0.326372849605006
    Attrib odor=s    0.4474641750563456
    Attrib odor=m    0.12455565899792641
    Attrib gill-attachment=a    -0.014120064432252171
    Attrib gill-spacing=w    -0.9869482485544792
    Attrib gill-size=b    -1.215009287949375
    Attrib gill-color=k    -0.11283786786197109
    Attrib gill-color=n    -0.12199328783304215
    Attrib gill-color=g    -0.06060378712058732
    Attrib gill-color=p    -0.5128267864722225
    Attrib gill-color=w    -0.09366951251308586
    Attrib gill-color=h    0.013674054233328894
    Attrib gill-color=u    -0.009081130160483104
    Attrib gill-color=e    -0.16141696581483178
    Attrib gill-color=b    0.8157544895678622
    Attrib gill-color=r    0.4894382340535653
    Attrib gill-color=y    0.06338820062209351
    Attrib gill-color=o    0.026300786110382158
    Attrib stalk-shape=t    -0.6965328986008118
    Attrib stalk-root=e    -0.23360653546217852
    Attrib stalk-root=c    -0.7941699664235812
    Attrib stalk-root=b    1.2211416894292555
    Attrib stalk-root=r    -0.41828646395590924
    Attrib stalk-surface-above-ring=s    -0.3545213459146379
    Attrib stalk-surface-above-ring=f    -0.24521960044214067
    Attrib stalk-surface-above-ring=k    0.7899910650604335
    Attrib stalk-surface-above-ring=y    -0.0995076861335569
    Attrib stalk-surface-below-ring=s    0.09862442180656877
    Attrib stalk-surface-below-ring=f    -0.5015844832399682
    Attrib stalk-surface-below-ring=y    0.3793300626976676
    Attrib stalk-surface-below-ring=k    -0.007947133026927206
    Attrib stalk-color-above-ring=w    0.2088300945512182
    Attrib stalk-color-above-ring=g    -0.11669023336035664
    Attrib stalk-color-above-ring=p    0.011980203390443473
    Attrib stalk-color-above-ring=n    -0.03563861561176282
    Attrib stalk-color-above-ring=b    0.11665302575847139
    Attrib stalk-color-above-ring=e    -0.12561615545145396
    Attrib stalk-color-above-ring=o    -0.11208711017718517
    Attrib stalk-color-above-ring=c    0.10038636605274033
    Attrib stalk-color-above-ring=y    0.01828842953615123
    Attrib stalk-color-below-ring=w    0.2911869994456425
    Attrib stalk-color-below-ring=p    0.05123914215013149
    Attrib stalk-color-below-ring=g    -0.3352612176392518
    Attrib stalk-color-below-ring=b    0.06584804907484267
    Attrib stalk-color-below-ring=n    0.0062003256581666475
    Attrib stalk-color-below-ring=e    -0.10494712827785709
    Attrib stalk-color-below-ring=y    0.19630052354012645
    Attrib stalk-color-below-ring=o    -0.0513634430937036
    Attrib stalk-color-below-ring=c    0.1415334580869417
    Attrib veil-type    -0.028298240232195515
    Attrib veil-color=w    0.06407757495896375
    Attrib veil-color=n    -0.03483330863947793
    Attrib veil-color=o    -0.014944866362838066
    Attrib veil-color=y    0.08410116468205386
    Attrib ring-number=o    -0.19666220272588394
    Attrib ring-number=t    0.1514796130164485
    Attrib ring-number=n    0.11035389359458324
    Attrib ring-type=p    -0.02485886901709342
    Attrib ring-type=e    0.29259767556080335
    Attrib ring-type=l    0.20632531844760466
    Attrib ring-type=f    -0.5336601265660079
    Attrib ring-type=n    0.16106326207494606
    Attrib spore-print-color=k    -0.35257207920283934
    Attrib spore-print-color=n    -1.0712980208503167
    Attrib spore-print-color=u    -0.6430057293320428
    Attrib spore-print-color=h    0.32471840430040655
    Attrib spore-print-color=w    0.10049077478509325
    Attrib spore-print-color=r    1.877021367630887
    Attrib spore-print-color=o    -0.04339274064081708
    Attrib spore-print-color=y    -0.023482416502608575
    Attrib spore-print-color=b    0.04783764636401737
    Attrib population=s    0.1518968381031098
    Attrib population=n    -0.37905578236259474
    Attrib population=a    -0.12427857555221133
    Attrib population=v    0.5470892766622345
    Attrib population=y    -0.5797557815773006
    Attrib population=c    0.453535938336061
    Attrib habitat=u    -0.07850519551529775
    Attrib habitat=g    0.48907159407719514
    Attrib habitat=m    0.4749693840000854
    Attrib habitat=d    -0.4478209330459248
    Attrib habitat=p    -0.14044999318595577
    Attrib habitat=w    -0.35217013829099747
    Attrib habitat=l    0.24939697843394815
Sigmoid Node 3
    Inputs    Weights
    Threshold    0.01657824906767966
    Attrib cap-shape=x    -0.12485746302370651
    Attrib cap-shape=b    0.05253123372813971
    Attrib cap-shape=s    -0.25890720332683814
    Attrib cap-shape=f    0.23689688544668092
    Attrib cap-shape=k    0.03485460507264443
    Attrib cap-shape=c    0.06445617374523117
    Attrib cap-surface=s    0.20605387629340974
    Attrib cap-surface=y    -0.12351566978506429
    Attrib cap-surface=f    -0.16977644694994418
    Attrib cap-surface=g    0.057304913490919374
    Attrib cap-color=n    -0.10188672381160073
    Attrib cap-color=y    -0.04228632144605792
    Attrib cap-color=w    0.01702550297870574
    Attrib cap-color=g    -0.041172337276242255
    Attrib cap-color=e    -0.07703710433295692
    Attrib cap-color=p    0.40803247182604363
    Attrib cap-color=b    0.17805025631497032
    Attrib cap-color=u    -0.061746166319855356
    Attrib cap-color=c    -0.16869211343682458
    Attrib cap-color=r    -0.22630741503695367
    Attrib bruises=f    7.612699998176052E-4
    Attrib odor=p    0.9875205376934404
    Attrib odor=a    -0.9734509100709802
    Attrib odor=l    -0.9672717786213408
    Attrib odor=n    -1.8613146098270636
    Attrib odor=f    1.175115079322973
    Attrib odor=c    0.9934202812384874
    Attrib odor=y    0.20765489712857083
    Attrib odor=s    0.13556742292579796
    Attrib odor=m    0.0803842608685126
    Attrib gill-attachment=a    -0.055598992746088316
    Attrib gill-spacing=w    -1.0586795305600043
    Attrib gill-size=b    -1.2914225137148265
    Attrib gill-color=k    -0.1509037135572638
    Attrib gill-color=n    -0.3778437257566047
    Attrib gill-color=g    0.021612734068803176
    Attrib gill-color=p    -0.11679827456498627
    Attrib gill-color=w    -0.39531161429063577
    Attrib gill-color=h    0.19486258933080483
    Attrib gill-color=u    -0.0035371990032267066
    Attrib gill-color=e    -0.12395153881374486
    Attrib gill-color=b    0.479450315549394
    Attrib gill-color=r    0.17951666356535023
    Attrib gill-color=y    0.18642742253227348
    Attrib gill-color=o    -0.0924610231281136
    Attrib stalk-shape=t    -0.5527684818111036
    Attrib stalk-root=e    -0.04064164536522284
    Attrib stalk-root=c    -0.25125050969795615
    Attrib stalk-root=b    0.6425865011259179
    Attrib stalk-root=r    -0.4427692861178643
    Attrib stalk-surface-above-ring=s    -0.4497611587103567
    Attrib stalk-surface-above-ring=f    -0.5082359667992117
    Attrib stalk-surface-above-ring=k    0.7449335328623359
    Attrib stalk-surface-above-ring=y    0.28947890844650676
    Attrib stalk-surface-below-ring=s    -0.09922978243270339
    Attrib stalk-surface-below-ring=f    -0.578065101990651
    Attrib stalk-surface-below-ring=y    0.6363719386065555
    Attrib stalk-surface-below-ring=k    0.026759606979505947
    Attrib stalk-color-above-ring=w    -0.1723753160873474
    Attrib stalk-color-above-ring=g    -0.21019916427867438
    Attrib stalk-color-above-ring=p    -0.010990259236326924
    Attrib stalk-color-above-ring=n    -0.07608936584523822
    Attrib stalk-color-above-ring=b    0.0800843560532248
    Attrib stalk-color-above-ring=e    -0.046703496609959684
    Attrib stalk-color-above-ring=o    -0.16187650462422065
    Attrib stalk-color-above-ring=c    0.08038736183872379
    Attrib stalk-color-above-ring=y    0.2856938840703906
    Attrib stalk-color-below-ring=w    -0.10325380323283323
    Attrib stalk-color-below-ring=p    0.05577319753094782
    Attrib stalk-color-below-ring=g    -0.25032889856248874
    Attrib stalk-color-below-ring=b    0.053437446849484906
    Attrib stalk-color-below-ring=n    -0.14689182561983063
    Attrib stalk-color-below-ring=e    -0.06336629670029181
    Attrib stalk-color-below-ring=y    0.5084109176351806
    Attrib stalk-color-below-ring=o    -0.12273377465526149
    Attrib stalk-color-below-ring=c    0.07835628722220105
    Attrib veil-type    -0.04012020665928117
    Attrib veil-color=w    -0.25912953603395966
    Attrib veil-color=n    -0.04156952461596815
    Attrib veil-color=o    -0.10107290717340468
    Attrib veil-color=y    0.3640169482496896
    Attrib ring-number=o    0.11247721509871102
    Attrib ring-number=t    -0.15604244394471808
    Attrib ring-number=n    0.09262197209368729
    Attrib ring-type=p    0.042159418193919974
    Attrib ring-type=e    0.12361298448881564
    Attrib ring-type=l    0.17625227286429926
    Attrib ring-type=f    -0.509125351922842
    Attrib ring-type=n    0.0676977847058554
    Attrib spore-print-color=k    -0.4897053827671778
    Attrib spore-print-color=n    -0.48825785250688686
    Attrib spore-print-color=u    -0.5083916156613691
    Attrib spore-print-color=h    0.5126418797162043
    Attrib spore-print-color=w    0.13782641036093402
    Attrib spore-print-color=r    0.7761674965761456
    Attrib spore-print-color=o    -0.07250360706425218
    Attrib spore-print-color=y    -0.05951500078256674
    Attrib spore-print-color=b    -0.023507939584285847
    Attrib population=s    0.14253314021333358
    Attrib population=n    -0.21797330358170583
    Attrib population=a    -0.14730168421256967
    Attrib population=v    0.2948804750224521
    Attrib population=y    -0.5203730044353285
    Attrib population=c    0.37995904517710916
    Attrib habitat=u    0.11470159959738485
    Attrib habitat=g    0.2055593529969377
    Attrib habitat=m    0.1046533322902553
    Attrib habitat=d    0.0389282766370988
    Attrib habitat=p    -0.19736124023877402
    Attrib habitat=w    -0.226687587548294
    Attrib habitat=l    -0.07029258200640559
Sigmoid Node 4
    Inputs    Weights
    Threshold    0.009390079269774584
    Attrib cap-shape=x    -0.21988276278879612
    Attrib cap-shape=b    0.3070599357096828
    Attrib cap-shape=s    -0.1042983445084498
    Attrib cap-shape=f    -0.018488572094864944
    Attrib cap-shape=k    -0.03833100008415889
    Attrib cap-shape=c    0.07826934856379193
    Attrib cap-surface=s    0.25322313143101344
    Attrib cap-surface=y    0.09716183631905251
    Attrib cap-surface=f    -0.43497948875928616
    Attrib cap-surface=g    0.08817459960089646
    Attrib cap-color=n    -0.23725462950899476
    Attrib cap-color=y    -0.17864946935440346
    Attrib cap-color=w    0.3361257700909337
    Attrib cap-color=g    -0.2522494615058385
    Attrib cap-color=e    0.10363855887481001
    Attrib cap-color=p    0.325483610152335
    Attrib cap-color=b    0.35410196375233266
    Attrib cap-color=u    -0.0408527572451095
    Attrib cap-color=c    -0.14622589776477207
    Attrib cap-color=r    -0.10607047535121171
    Attrib bruises=f    -0.010433590092996855
    Attrib odor=p    0.513001975008763
    Attrib odor=a    -0.6924499005835814
    Attrib odor=l    -0.6945036508211421
    Attrib odor=n    -1.2478276790406972
    Attrib odor=f    0.9303363537604853
    Attrib odor=c    0.4364990046334033
    Attrib odor=y    0.32782872568242444
    Attrib odor=s    0.39386317539920446
    Attrib odor=m    0.06407603340967907
    Attrib gill-attachment=a    -0.02141407300152443
    Attrib gill-spacing=w    -1.0163218185519718
    Attrib gill-size=b    -0.6021137245461783
    Attrib gill-color=k    -0.17643119258254594
    Attrib gill-color=n    -0.14116986173314658
    Attrib gill-color=g    0.09816638501273534
    Attrib gill-color=p    -0.4009149850044336
    Attrib gill-color=w    -0.43230876467361806
    Attrib gill-color=h    0.05368542145782693
    Attrib gill-color=u    -0.020902390537278926
    Attrib gill-color=e    -0.15365143738806503
    Attrib gill-color=b    0.8336593482883076
    Attrib gill-color=r    0.6887331628778948
    Attrib gill-color=y    0.05077902290295779
    Attrib gill-color=o    -0.04990945567531309
    Attrib stalk-shape=t    -0.1233608108106777
    Attrib stalk-root=e    -0.09627578789416402
    Attrib stalk-root=c    -0.49233775299045784
    Attrib stalk-root=b    0.8516349140832816
    Attrib stalk-root=r    -0.17151728604021685
    Attrib stalk-surface-above-ring=s    -0.41434388357609414
    Attrib stalk-surface-above-ring=f    -0.1084733958782109
    Attrib stalk-surface-above-ring=k    0.5869714146616822
    Attrib stalk-surface-above-ring=y    -0.0695561912692735
    Attrib stalk-surface-below-ring=s    -0.187081422684397
    Attrib stalk-surface-below-ring=f    -0.09854609456396726
    Attrib stalk-surface-below-ring=y    0.17452517250927047
    Attrib stalk-surface-below-ring=k    0.24317083138860343
    Attrib stalk-color-above-ring=w    -0.0246131009948354
    Attrib stalk-color-above-ring=g    -0.11325133805722992
    Attrib stalk-color-above-ring=p    0.1423918383176478
    Attrib stalk-color-above-ring=n    -0.01614054426756081
    Attrib stalk-color-above-ring=b    0.07202393026484374
    Attrib stalk-color-above-ring=e    -0.012251999761561197
    Attrib stalk-color-above-ring=o    -0.013570447248375084
    Attrib stalk-color-above-ring=c    0.14393300556522007
    Attrib stalk-color-above-ring=y    0.01620284661591453
    Attrib stalk-color-below-ring=w    0.09635930340381238
    Attrib stalk-color-below-ring=p    0.13257835901509207
    Attrib stalk-color-below-ring=g    -0.2723937629671323
    Attrib stalk-color-below-ring=b    0.128772191746333
    Attrib stalk-color-below-ring=n    -0.010962927616345116
    Attrib stalk-color-below-ring=e    -0.08160874300927495
    Attrib stalk-color-below-ring=y    0.06143994140181253
    Attrib stalk-color-below-ring=o    -0.043325042388568796
    Attrib stalk-color-below-ring=c    0.13588662441254798
    Attrib veil-type    -0.03732987298031826
    Attrib veil-color=w    0.006732987665661073
    Attrib veil-color=n    0.024730632456152608
    Attrib veil-color=o    -0.019663730496613827
    Attrib veil-color=y    0.01673764888899056
    Attrib ring-number=o    -0.4320313196607484
    Attrib ring-number=t    0.2714926457385527
    Attrib ring-number=n    0.0835904643982606
    Attrib ring-type=p    -0.3530387539020081
    Attrib ring-type=e    0.32896402011583165
    Attrib ring-type=l    0.2221179659825829
    Attrib ring-type=f    -0.31803922029943826
    Attrib ring-type=n    0.1520130105793839
    Attrib spore-print-color=k    -0.4863259630077061
    Attrib spore-print-color=n    -0.9145277280504769
    Attrib spore-print-color=u    -0.22165757114272222
    Attrib spore-print-color=h    0.5110068902665951
    Attrib spore-print-color=w    -0.3297282697330297
    Attrib spore-print-color=r    1.7079128531071976
    Attrib spore-print-color=o    0.007419379518699029
    Attrib spore-print-color=y    0.010467787322905367
    Attrib spore-print-color=b    -0.007986707254438124
    Attrib population=s    -0.13542277741446285
    Attrib population=n    -0.17337269609930925
    Attrib population=a    -0.07669626872841645
    Attrib population=v    0.8303866747111571
    Attrib population=y    -0.34584471102435405
    Attrib population=c    0.016622366636330488
    Attrib habitat=u    0.07081739058999334
    Attrib habitat=g    0.3656547567108462
    Attrib habitat=m    0.6865841429401518
    Attrib habitat=d    -0.3899661172271507
    Attrib habitat=p    -0.25221558820953427
    Attrib habitat=w    -0.18390870721653396
    Attrib habitat=l    -0.024309627534031004
Sigmoid Node 5
    Inputs    Weights
    Threshold    -0.03658945529023487
    Attrib cap-shape=x    -0.1925962486457505
    Attrib cap-shape=b    0.14108203318673493
    Attrib cap-shape=s    -0.20822631395459693
    Attrib cap-shape=f    -0.07074650115749984
    Attrib cap-shape=k    -0.011799869218070887
    Attrib cap-shape=c    0.3874970669513733
    Attrib cap-surface=s    -0.28507547904929825
    Attrib cap-surface=y    -0.08010373306288374
    Attrib cap-surface=f    -0.17579233809810257
    Attrib cap-surface=g    0.5184511669792703
    Attrib cap-color=n    -0.26852953626366316
    Attrib cap-color=y    0.0792707055023752
    Attrib cap-color=w    0.40420895576979854
    Attrib cap-color=g    -0.0946537517912806
    Attrib cap-color=e    -0.050723905709296316
    Attrib cap-color=p    0.08474573547372256
    Attrib cap-color=b    0.35188912047258847
    Attrib cap-color=u    -0.06595207884877254
    Attrib cap-color=c    -0.36503915594689906
    Attrib cap-color=r    -0.12045733622389007
    Attrib bruises=f    -0.21856509809567443
    Attrib odor=p    0.8491944956220832
    Attrib odor=a    -0.9289736212689109
    Attrib odor=l    -1.1137293686754828
    Attrib odor=n    -1.170400531575971
    Attrib odor=f    0.9376969959198231
    Attrib odor=c    0.9752101025051726
    Attrib odor=y    0.3374150402404221
    Attrib odor=s    0.2467521518430756
    Attrib odor=m    0.12172222843611528
    Attrib gill-attachment=a    -0.08417981392653516
    Attrib gill-spacing=w    -0.8252615639155171
    Attrib gill-size=b    -1.4425029544431054
    Attrib gill-color=k    -0.14731273216055357
    Attrib gill-color=n    -0.19219310060099654
    Attrib gill-color=g    -0.11484624265473617
    Attrib gill-color=p    -0.25162883930674596
    Attrib gill-color=w    -0.007555566593464481
    Attrib gill-color=h    0.027579392027685192
    Attrib gill-color=u    -0.05344107584547652
    Attrib gill-color=e    -0.14848074983316728
    Attrib gill-color=b    0.6105444485551735
    Attrib gill-color=r    0.2612728918243285
    Attrib gill-color=y    0.09165203222371185
    Attrib gill-color=o    0.01417760196365611
    Attrib stalk-shape=t    -0.6872799387563817
    Attrib stalk-root=e    0.005193020408197468
    Attrib stalk-root=c    -0.6057957714931439
    Attrib stalk-root=b    0.8096625968100886
    Attrib stalk-root=r    -0.3921966039782357
    Attrib stalk-surface-above-ring=s    -0.3418412208979699
    Attrib stalk-surface-above-ring=f    -0.4250393242116298
    Attrib stalk-surface-above-ring=k    0.684391427401616
    Attrib stalk-surface-above-ring=y    0.04264075830394366
    Attrib stalk-surface-below-ring=s    -0.1803982127276327
    Attrib stalk-surface-below-ring=f    -0.4170934111169475
    Attrib stalk-surface-below-ring=y    0.6017775099035786
    Attrib stalk-surface-below-ring=k    0.016145844426058837
    Attrib stalk-color-above-ring=w    0.022512094304991404
    Attrib stalk-color-above-ring=g    -0.12729220083171344
    Attrib stalk-color-above-ring=p    0.11969810490278669
    Attrib stalk-color-above-ring=n    -0.03178323693933211
    Attrib stalk-color-above-ring=b    0.10550448823886834
    Attrib stalk-color-above-ring=e    -0.14448738309977766
    Attrib stalk-color-above-ring=o    -0.05645232578433177
    Attrib stalk-color-above-ring=c    0.14208986353590863
    Attrib stalk-color-above-ring=y    0.1393349701253175
    Attrib stalk-color-below-ring=w    0.2687264555225265
    Attrib stalk-color-below-ring=p    0.02479695559477557
    Attrib stalk-color-below-ring=g    -0.14370720994914854
    Attrib stalk-color-below-ring=b    0.07709986701640217
    Attrib stalk-color-below-ring=n    -0.3864701852409661
    Attrib stalk-color-below-ring=e    -0.136932884053782
    Attrib stalk-color-below-ring=y    0.29959522239530123
    Attrib stalk-color-below-ring=o    -0.08970626001214405
    Attrib stalk-color-below-ring=c    0.09023309830159372
    Attrib veil-type    0.0034665785791291315
    Attrib veil-color=w    -0.07320394346745403
    Attrib veil-color=n    -0.06663085775793659
    Attrib veil-color=o    -0.054894508752531865
    Attrib veil-color=y    0.21126065878007902
    Attrib ring-number=o    0.08787100291313024
    Attrib ring-number=t    -0.24135553698663842
    Attrib ring-number=n    0.06387345096944176
    Attrib ring-type=p    0.2138048310358505
    Attrib ring-type=e    -0.031287455859804476
    Attrib ring-type=l    0.26490819720679315
    Attrib ring-type=f    -0.5232428122388354
    Attrib ring-type=n    0.1283752634096102
    Attrib spore-print-color=k    -0.42296747806953805
    Attrib spore-print-color=n    -0.6421806912116764
    Attrib spore-print-color=u    -0.5799865256483921
    Attrib spore-print-color=h    0.3135575321141772
    Attrib spore-print-color=w    0.13510486062309718
    Attrib spore-print-color=r    1.3096942763936974
    Attrib spore-print-color=o    -0.03298225828817847
    Attrib spore-print-color=y    -0.004923032753215338
    Attrib spore-print-color=b    -0.04443376164689844
    Attrib population=s    0.05717018459128891
    Attrib population=n    -0.33586856409084803
    Attrib population=a    -0.08719618293012596
    Attrib population=v    -0.08671466193443178
    Attrib population=y    -0.4329579379767514
    Attrib population=c    0.9313150416136873
    Attrib habitat=u    0.12959011416018204
    Attrib habitat=g    0.15852664110927545
    Attrib habitat=m    0.3192305013832028
    Attrib habitat=d    -0.13425119403511146
    Attrib habitat=p    -0.37629285206508495
    Attrib habitat=w    -0.39133741240554215
    Attrib habitat=l    0.37231059066397293
Sigmoid Node 6
    Inputs    Weights
    Threshold    0.050156871548619544
    Attrib cap-shape=x    0.18379705395150822
    Attrib cap-shape=b    -0.2136414792028981
    Attrib cap-shape=s    -0.2802791838616005
    Attrib cap-shape=f    -0.06518853169188114
    Attrib cap-shape=k    0.1376060188515848
    Attrib cap-shape=c    0.12616550222284062
    Attrib cap-surface=s    0.14483340241545742
    Attrib cap-surface=y    -0.18775848918355606
    Attrib cap-surface=f    -0.15998357043620204
    Attrib cap-surface=g    0.29354855005194347
    Attrib cap-color=n    -0.3269187996812452
    Attrib cap-color=y    0.0716059087207979
    Attrib cap-color=w    0.4348075061891272
    Attrib cap-color=g    0.05728442386075474
    Attrib cap-color=e    0.04923309823134951
    Attrib cap-color=p    0.19735518803175015
    Attrib cap-color=b    0.06736630482664586
    Attrib cap-color=u    -0.04105452072513363
    Attrib cap-color=c    -0.3153071397485994
    Attrib cap-color=r    -0.13287889854086632
    Attrib bruises=f    0.02109132404900671
    Attrib odor=p    1.2771267603317635
    Attrib odor=a    -1.129777060039362
    Attrib odor=l    -1.1502447208491202
    Attrib odor=n    -2.0420334370890143
    Attrib odor=f    1.204375872820077
    Attrib odor=c    1.0959350317500816
    Attrib odor=y    0.2685339786151729
    Attrib odor=s    0.3296776473343314
    Attrib odor=m    0.091106674017105
    Attrib gill-attachment=a    -0.07902844763011832
    Attrib gill-spacing=w    -0.923428928504284
    Attrib gill-size=b    -1.9850018053557146
    Attrib gill-color=k    0.04852622150749068
    Attrib gill-color=n    -0.3309817559992028
    Attrib gill-color=g    -0.43328438678691544
    Attrib gill-color=p    0.029444471054344275
    Attrib gill-color=w    -0.12045939604607868
    Attrib gill-color=h    0.02680446555563995
    Attrib gill-color=u    0.03385380716901403
    Attrib gill-color=e    -0.15497368447610876
    Attrib gill-color=b    0.7235062109984443
    Attrib gill-color=r    0.20272779632454618
    Attrib gill-color=y    0.17122178743515482
    Attrib gill-color=o    0.004465696618834776
    Attrib stalk-shape=t    -0.343523890252838
    Attrib stalk-root=e    0.38261137632439995
    Attrib stalk-root=c    -0.3470373499782265
    Attrib stalk-root=b    0.5560521946237038
    Attrib stalk-root=r    -0.528079840855353
    Attrib stalk-surface-above-ring=s    -0.8099228247315731
    Attrib stalk-surface-above-ring=f    -0.4827479891659312
    Attrib stalk-surface-above-ring=k    1.1063209826657634
    Attrib stalk-surface-above-ring=y    0.22534437872370194
    Attrib stalk-surface-below-ring=s    -0.20589594409424655
    Attrib stalk-surface-below-ring=f    -0.5368901130379895
    Attrib stalk-surface-below-ring=y    0.7535775099564285
    Attrib stalk-surface-below-ring=k    -0.0677205124750027
    Attrib stalk-color-above-ring=w    -0.13868015926208996
    Attrib stalk-color-above-ring=g    -0.16244445775780536
    Attrib stalk-color-above-ring=p    0.03853546509872221
    Attrib stalk-color-above-ring=n    -0.0024999265147452113
    Attrib stalk-color-above-ring=b    0.12001626065992041
    Attrib stalk-color-above-ring=e    -0.12966231041332138
    Attrib stalk-color-above-ring=o    -0.08584649635902071
    Attrib stalk-color-above-ring=c    0.07363297233573428
    Attrib stalk-color-above-ring=y    0.24418916953949235
    Attrib stalk-color-below-ring=w    0.1736012150931874
    Attrib stalk-color-below-ring=p    -0.050166474084039944
    Attrib stalk-color-below-ring=g    -0.3306723608088944
    Attrib stalk-color-below-ring=b    0.04140920929161635
    Attrib stalk-color-below-ring=n    -0.3357965457390945
    Attrib stalk-color-below-ring=e    -0.02467784610713902
    Attrib stalk-color-below-ring=y    0.4257860698176422
    Attrib stalk-color-below-ring=o    -0.1243839119957078
    Attrib stalk-color-below-ring=c    0.11508020777589538
    Attrib veil-type    -0.006432707254290203
    Attrib veil-color=w    -0.15831766586964088
    Attrib veil-color=n    -0.07340402436246017
    Attrib veil-color=o    -0.027742174518289787
    Attrib veil-color=y    0.3152028566064354
    Attrib ring-number=o    0.5607768513563499
    Attrib ring-number=t    -0.718835041528303
    Attrib ring-number=n    0.14280824248776644
    Attrib ring-type=p    0.12261375705655819
    Attrib ring-type=e    0.23759198104221893
    Attrib ring-type=l    0.24368972691946675
    Attrib ring-type=f    -0.5802117272878724
    Attrib ring-type=n    0.12683333155486773
    Attrib spore-print-color=k    -0.3086063859058305
    Attrib spore-print-color=n    -0.6450085698912341
    Attrib spore-print-color=u    -0.6005991987540561
    Attrib spore-print-color=h    0.536578016238678
    Attrib spore-print-color=w    0.8283123739011373
    Attrib spore-print-color=r    0.32562807963456447
    Attrib spore-print-color=o    -0.055595077228341346
    Attrib spore-print-color=y    -0.05104519097364698
    Attrib spore-print-color=b    -0.040504545595954435
    Attrib population=s    0.44291783725606115
    Attrib population=n    -0.3521853472914256
    Attrib population=a    -0.1691700233034546
    Attrib population=v    -0.24648427602564643
    Attrib population=y    -0.6325690905689909
    Attrib population=c    0.9113508511987067
    Attrib habitat=u    0.1051416591674923
    Attrib habitat=g    -0.042371478263748
    Attrib habitat=m    -0.020896705481773546
    Attrib habitat=d    0.009811149176744253
    Attrib habitat=p    0.15657678951775644
    Attrib habitat=w    -0.3427584306202756
    Attrib habitat=l    0.048740390326428455
Sigmoid Node 7
    Inputs    Weights
    Threshold    2.999199506783947
    Node 2    -1.5037192721620785
    Node 3    -1.8099123888870787
    Node 4    -1.9510555730877615
    Node 5    -2.3680477988748483
    Node 6    -2.1242787938461167
Sigmoid Node 8
    Inputs    Weights
    Threshold    3.0004131801280223
    Node 2    -1.4665660299174827
    Node 3    -1.821031490769671
    Node 4    -2.002785924579732
    Node 5    -2.3978195601497356
    Node 6    -2.1079897176363915
Sigmoid Node 9
    Inputs    Weights
    Threshold    2.8789349729522495
    Node 2    -1.4632491155649858
    Node 3    -1.7618492453501449
    Node 4    -1.9113050501188336
    Node 5    -2.2760508616360777
    Node 6    -2.041467468146679
Class p
    Input
    Node 0
Class e
    Input
    Node 1

=== Confusion Matrix ===

    a    b   <-- classified as
 3916    0 |    a = p
    0 4208 |    b = e
         */


        Evaluation ev = new Evaluation(dane);
        ev.evaluateModel(mp, dane);
        Matrix.show(ev.confusionMatrix());
/*
3916    0
   0 4208
*/

        Instances dane_pomieszane = new Instances(dane2);
        dane_pomieszane.randomize(new Random(0));
        int wielkoscTestowego = 1000;
        Instances dp_test = new Instances(dane_pomieszane, wielkoscTestowego);
        Instances dp_train = new Instances(dane_pomieszane, dane_pomieszane.size()-wielkoscTestowego);
        for(int i = 0; i<dane_pomieszane.size(); i++) {
            if (i < wielkoscTestowego) {
                dp_test.add(dane_pomieszane.get(i));
            }
            else {
                dp_train.add(dane_pomieszane.get(i));
            }
        }
        System.out.println(dp_test.numInstances());
        System.out.println("Testowe: ");
        System.out.println(dp_test);
        System.out.println(dp_train.numInstances());
        System.out.println("Treningowe: ");
        System.out.println(dp_train);

        MultilayerPerceptron mp2 = new MultilayerPerceptron();
        mp.setHiddenLayers("5,3");
        mp.setLearningRate(0.3);
        //mp.setGUI(true);
        mp.buildClassifier(dp_train);
        System.out.println(mp);

        Evaluation ev2 = new Evaluation(dp_test);
        ev2.evaluateModel(mp, dp_test);
        Matrix.show(ev2.confusionMatrix());
//        477   0
//        0 523
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
