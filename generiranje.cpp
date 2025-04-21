#include <iostream>
#include <random>
#include <ctime>
#include<utility>
#include<fstream>
#include <algorithm>
#include <map>
#include <chrono>
#include <iomanip>
using namespace std;
typedef pair<int,int> pii;

#include "problem-sets.h"
int n;

vector< pair<pii,int> > augmenting_edges, edges;

vector<vector<int> > sus;

vector<int> depth;
vector<int> par;

vector<vector<int>> M;//(n-1, augmenting_edges.size() )
vector<vector<int>> retci, stupci;
vector<bool> pokriven;

int n_datoteka, e_datoteka;
vector<int> ei1,ei2,w;
vector<int> chosen;
vector<int> znacajke1, znacajke2;

void dfs( int cur , int dub ){
	
	depth[cur]=dub;
	
	for(int i=0;i<sus[cur].size();i++){
		int z=sus[cur][i];
		
		if( z==par[cur] )continue;
		
		par[z]=cur;
		dfs( z , dub+1 );
	}
	return;
}


void init(){
	
	depth.resize(n);
	par.resize(n);
	pokriven.resize(n,false);
	
	M.resize(n-1);
	retci.resize( n-1 ); stupci.resize( augmenting_edges.size() );
	
	for(int i=0;i<n-1;i++){
		M[i].resize(augmenting_edges.size(),0);//augmenting_edges.size()
	}
	
	par[0]=-1;
	dfs( 0 , 0 );
	
	return;
}

/*

ARHITEKTURA FAJLA:
(e je broj usmjerenih edgeva, svaki T-T par predstavlja dva usmjerena brida kao npr 3-17 , 17-3 )
n e
9 58
ei1
T T T T T T T T A A A A A A A A A A A A A A A A A A A A A
ei2
T T T T T T T T A A A A A A A A A A A A A A A A A A A A A
weights
0 0 0 0 0 0 0 0 w w w w w w w w w w w w w w w w w w w w w
chosen (prvih n-1 je u pocetnom stablu)
1 1 1 1 1 1 1 1|0 0 1 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 1 0 0
feature1 feature2 , feature1=broj susjeda, feature2=broj susjeda susjedova
number_11 number_12
number_21 number_22
...
number_n1 number_n2
*/

void graf_u_datoteku(int a, int b, int c){
	ei1.push_back(a);	ei1.push_back(b);
	ei2.push_back(b);	ei2.push_back(a);
	w.push_back(c);		w.push_back(c);
}

/*
Kruskalov algoritam

neki bridovi budu dodani u mst dok oni preostali budu dodani u 
skup bridova koje cemo dodavati ne bi li makli sve mostove "augmenting_edges"
*/
void mst( vector< pair<pii,int> >& edges, vector< pair<pii,int> >& augmenting_edges ){
	
	n_datoteka=n; e_datoteka=edges.size()*2;
	chosen.resize( 2*( edges.size() - (n-1) ), 0 );
	
	vector< pair<pii,int> > T, A;
	
	vector<int> pripadnost;
	vector< vector<int> > grupa;
	pripadnost.resize(n), sus.resize(n), grupa.resize(n);
	for(int i=0;i<n;i++)pripadnost[i]=i, grupa[i].push_back(i);
	
	// Sorting using lambda function
    sort(edges.begin(), edges.end(), [](const pair<pii, int> &L, const pair<pii, int> &R) {
        return L.second < R.second;
    });
	
	for( auto e:edges ){
		int a=e.first.first;
		int b=e.first.second;
		
		if(grupa[a].size() < grupa[b].size())swap(a,b);
		
		if(pripadnost[a]!=pripadnost[b]){//ovi budu cinili graf ko u generiranje.cpp
			
			sus[a].push_back(b);
			sus[b].push_back(a);
			
			int iz=pripadnost[b];
			int u=pripadnost[a];
			
			for( auto clan:grupa[iz] ){
				pripadnost[clan]=u;
				grupa[u].push_back( clan );
			}
			grupa[iz].clear();
			T.push_back( e );//edge weight bude 0
		}
		else{
			augmenting_edges.push_back( e );
			A.push_back( e );
		}
	}
	
	
	
	//datotekaispis
	for( auto e : T )graf_u_datoteku( e.first.first, e.first.second, 0 );//0 namjerno-vazno, ovo bi mogo maknut da se ne jebem s glupostima
	for( auto e : A )graf_u_datoteku( e.first.first, e.first.second, e.second );
	
}


void scip_u_cpp() {
    string fileName = "solution.log";
    ifstream inputFile(fileName);
    
    if (!inputFile.is_open()) {
        cerr << "Error: Could not open the file " << fileName << endl;
        exit(1);
    }
    
    string line;
    bool pocelo=false;
    
    while (getline(inputFile, line)) {
        if (line.rfind("objective value:", 0) == 0) {
//            cout << "Found the line: " << line << endl;
            pocelo=true;
            continue;
        }
        if(pocelo){
        	if( line.length()<2 )break;
//        	cout<<line<<endl;
        	int broj=0;
        	for(int i=1;'0'<=line[i] && line[i]<='9';i++){
        		broj*=10;
        		broj+=line[i]-'0';
			}
			chosen[2*broj]=1;//dodo sam 2*broj ovdje da odgovara strukturi filea jer se inace potrga naprosto
			chosen[2*broj+1]=1;
//			cout<<broj<<endl;
		}
    }
    
    inputFile.close();
    return;
}


/*
za edge izmedu cvorova (a,b) vraca se koji svi edgeovi su prekriveni
*/
vector<int> cover( int a , int b ){
	if(depth[a]>depth[b])swap(a,b);
	
	vector<int> ret;
	
	while(depth[a]<depth[b]){
		pokriven[b]=true;
		ret.push_back(b);
		b=par[b];
	}
	
	while(a!=b){
		pokriven[a]=pokriven[b]=true;
		
		ret.push_back(a);
		a=par[a];
		
		ret.push_back(b);
		b=par[b];
	}
	
	return ret;
}




/*
stvara SCIP rjesenje problema nula indeksirano
*/
void SCIPispis(  ){//njemu su sam bitnim augmenting edges za set covering stvar

//int n, int e, vector< pair<pii,int> > augmenting_edges, vector<vector<int>> M

	ofstream outFile("set_cover_problem.lp");//set_cover_problem.lp
	// Redirect cout to the file
    streambuf* coutBuf = cout.rdbuf();  // Save old buffer
    cout.rdbuf(outFile.rdbuf());  // Redirect cout to outFile
	
	cout<<"Minimize"<<endl;
	cout<<" obj:";
	for(int i=0;i<augmenting_edges.size();i++){
		if(i)cout<<" +";
		cout<<" "<<augmenting_edges[i].second<<" "<<"e"<<to_string(i);
	}cout<<endl;
	
	cout<<"Subject To"<<endl;
	for(int j=0;j<n-1;j++){
		cout<<" c"<<to_string(j+1)<<":";
		
		int kol=0;
		
		for(int i=0;i<augmenting_edges.size();i++){
			if(M[j][i]){
				if(kol)cout<<" +";
				cout<<" e"<<to_string(i);
				kol++;
			}
		}
		
		cout<<" >= 1"<<endl;
	}
	
	cout<<"Bounds"<<endl;
	for(int i=0;i<augmenting_edges.size();i++){
		cout<<" 0 <= e"<<to_string(i)<<" <= 1"<<endl;
	}
	
	cout<<"General"<<endl;
	for(int i=0;i<augmenting_edges.size();i++){
		cout<<" e"<<to_string(i)<<endl;
	}
	cout<<"End"<<endl;
	
	// Restore cout's original buffer
    cout.rdbuf(coutBuf);

    // Close the file
    outFile.close();
	
	return;
}


void greedy_cover(){//vector<vector<int>> M;//( n-1, augmenting_edges.size() )
	
	// Get the starting time point
    auto start = chrono::high_resolution_clock::now();

	int covered[n]={0};//O( E' * N * log N )
	int kolko_pokriveno[n]={0};
	
	int kol[augmenting_edges.size()]={0};
	
	for(int i=0;i<augmenting_edges.size();i++){
		kol[i]=stupci[i].size();
	}
	
	vector<pii> rjesenje;
	
	for(int i=0;i<=n-2;i++){
		
		if(covered[i])continue;
		
		int ind=retci[i][0];
		
		double naj = (double) augmenting_edges[ind].second / kol[ind];
		
		for(int j=1;j<retci[i].size();j++){
			
			int tko=retci[i][j];
			
			if( naj > (double) augmenting_edges[tko].second / kol[tko] ){
				naj = (double) augmenting_edges[tko].second / kol[tko];
				ind = tko;
			}	
		}
		
		
		for(int j=0;j<stupci[ind].size();j++){
			
			int red=stupci[ind][j];
			kolko_pokriveno[red]++;
			
			if( covered[red] )continue;
			
			covered[red]=1;
			
			for(int g=0;g<retci[red].size();g++){
				int tko=retci[red][g];
				kol[tko]--;
			}
			
		}
		
		rjesenje.push_back(pii( -augmenting_edges[ind].second , ind  ));
	}
	
	int reza=0;
	
	sort(rjesenje.begin(),rjesenje.end());
	
	
	for(int i=0;i<rjesenje.size();i++){
		int tko=rjesenje[i].second;
		
		int cena=-rjesenje[i].first;
		
		int ok=1;
		
		for(int j=0;j<stupci[tko].size();j++){
			int red=stupci[tko][j];
			if(kolko_pokriveno[red]==1){
				ok=0;
				break;
			}
		}
		
		if(!ok){
			reza+=cena;
			continue;
		}
		else{
			for(int j=0;j<stupci[tko].size();j++){
				int red=stupci[tko][j];
				kolko_pokriveno[red]--;
			}
		}
	}
	
//	cout<<"greedy result:"<<reza<<endl;
	
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
//    cout << "Time taken by greedy_cover: " << fixed << setprecision(2) << duration.count() << " seconds" << endl;
}


void izracunaj_znacajke(){
	znacajke1.resize(n);
	znacajke2.resize(n);
	
	for(int i=0;i<n;i++)znacajke1[i]=sus[i].size();
	for(int i=0;i<n;i++){
		int suma=0;
		for(int j=0;j<sus[i].size();j++){
			suma+=sus[ sus[i][j] ].size();
		}
		znacajke2[i] = (int) suma/sus[i].size();
	}
	
}


bool prviput=true;
bool validation=false;
void cpp_u_python() {
    izracunaj_znacajke();
    
    string filename;
    if(!validation)filename = "data4python.txt";
    else filename = "validation.txt";
    
    if (prviput && ifstream(filename)) {  // File exists if this condition is true
        if (remove(filename.c_str()) != 0) {  // Delete the file
            cerr << "Error deleting existing file.\n";
        } else {
//            cout << "Existing file deleted successfully.\n";
        }
    }

    ofstream outFile(filename, ios::app);  // Open file in append mode
    if (outFile.is_open()) {
        outFile << n_datoteka << " " << e_datoteka << "\n";
        for (int i = 0; i < ei1.size(); i++) outFile << ei1[i] << " ";
        outFile << "\n";
        for (int i = 0; i < ei2.size(); i++) outFile << ei2[i] << " ";
        outFile << "\n";
        for (int i = 0; i < w.size(); i++) outFile << w[i] << " ";
        outFile << "\n";
        for (int i = 0; i < chosen.size(); i++) outFile << chosen[i] << " ";
        outFile << "\n";
        for (int i = 0; i < n; i++) outFile << znacajke1[i] << " " << znacajke2[i] << "\n";
        outFile.close();
//        cout << "Vector successfully appended to " + filename + "\n";
    } else {
        cerr << "Unable to open the file!";
    }
    
    
    if(validation){
	    int solution=0;
	    for(int i=0;i<chosen.size();i+=2){
	    	if(chosen[i]){
	    		solution+=w[i+2*(n-1)];
			}
		}
		
		string cover_filename = "coverage.txt";
	    if (prviput && ifstream(cover_filename)) {  // File exists if this condition is true
	    	
	        if (remove(cover_filename.c_str()) != 0) {  // Delete the file
	            cerr << "Error deleting existing file.\n";
	        } else {
//	            cout << "Existing file deleted successfully.\n";
	        }
	    }
	
	    ofstream outCoverage(cover_filename, ios::app);  // Open file in append mode
	    if (outCoverage.is_open()) {
	        outCoverage << augmenting_edges.size() << " " << solution << "\n";
	        for( auto vek : stupci){
	        	for( auto pokriven : vek ){
	        		outCoverage << pokriven << " ";
				}outCoverage << "\n";
			}
	        outCoverage.close();
//	        cout << "Vector successfully appended to " + cover_filename + "\n";
	    } else {
	        cerr << "Unable to open the file!";
	    }
	}
	
	prviput=false;
	
	return;
}


void stvori_jedan_primjer(){
	
	bool uniform_cost_edges=true;
	unif(edges, n, uniform_cost_edges);
//	euc(edges, n, uniform_cost_edges);
//	smallworld(edges, n, uniform_cost_edges);
	
	mst(edges, augmenting_edges );
	
	init();
	
	for(int i=0;i<augmenting_edges.size();i++){
		
		vector<int> ret = cover( augmenting_edges[i].first.first , augmenting_edges[i].first.second );
		
		for(auto cvor:ret){
			/*
				edge se zove po dijetetu, djeca su sva po velicini>=1,
				stoga preko cvor-1 guramo sve u nulti redak matrice M.
				Takoder matrica je dimenzija [n-1][augmenting_edges.size()]
			*/
			M[cvor-1][i]=1;
			retci[ cvor-1 ].push_back( i );
			stupci[ i ].push_back( cvor-1 );
		}
		
	}
	
	/*
	provjera pokrivenosti svih bridova s trenutnim skupom bridova
	jer mozda je nemoguce maknuti sve mostove s generiranim skupom bridova i test primjer stoga ne valja
	*/
	int broj_edgeva_stabla=n-1, broj_pokrivenih_edgeva=0;
	for(int i=1;i<n;i++){
		if(pokriven[i]){
			broj_pokrivenih_edgeva++;
		}
	}
	
//	cout<<"svi/pokriveni edgevi: " << broj_edgeva_stabla<<" "<<broj_pokrivenih_edgeva<<endl;
	
	if(broj_edgeva_stabla!=broj_pokrivenih_edgeva){
		cout<<"nije sve pokriveno lol"<<endl;
		return;
	}
	
	greedy_cover();
	SCIPispis();
	
	
	// Delete the solution.log file if it exists
    system("del solution.log > nul 2>&1");//ovi nulovi su da se ne ispisuje tekst meni u facu

    // Run SCIP with the given parameters
    system("scip -s params.set -f set_cover_problem.lp -l solution.log > nul 2>&1");

    // Optionally, print the content of solution.log (depends on your OS and file viewer)
    //system("solution.log");
    
    scip_u_cpp();
    
//    cout<<n_datoteka<<" "<<e_datoteka<<endl;
//    cout<<chosen.size()<<" "<<edges.size()<<" "<<endl;
//    cout<<ei1.size()<<" "<<w.size()<<endl;
	
	cpp_u_python();
	return;
}

void ciscenje_vektora(){
	// edgeovi
    augmenting_edges.clear();edges.clear();

    // struktura za edge covering
    sus.clear();M.clear();
    retci.clear();stupci.clear();

    // vektori za dfsove
    depth.clear(); par.clear(); pokriven.clear();

    // vektori za ispis
    ei1.clear(); ei2.clear();
    w.clear(); chosen.clear();
    znacajke1.clear(); znacajke2.clear();
}

int main(){
	
	srand(time(NULL));
	
	int t=1;//broj grafova u datasetu
	
	n=15;//broj cvorova u svakom grafu (trebo bih ovo napravit da je varijabilno ubuduce)
	
	for(int i=0;i<t;i++){
		if(i%100==0){
			cout<<"generirano grafova: "<<i<<endl;
		}
		ciscenje_vektora();
		validation=true;
		stvori_jedan_primjer();
	}
	
	cout<<"Primjeri uspjesno generirani!"<<endl;//shinzo abe
	
	return 0;
}

/*

znacajke 

ARHITEKTURA FAJLA koji definira graf:
(e je broj usmjerenih edgeva, svaki T-T par predstavlja dva usmjerena brida kao npr 3-17 , 17-3 )
VAZNO: napominjem ponovo da svaki JEDAN element ovdje u svakom retku se odnosi na dva smjera istog brida
i to mora tak biti zbog efikasne arhitekture machine learninga nad grafovima
(tezine i chosenost su samo za augmenting edgeove)
n e
9 58
ei1 (ei kao edge index)
T T T T T T T T A A A A A A A A A A A A A A A A A A A A A
ei2
T T T T T T T T A A A A A A A A A A A A A A A A A A A A A
weights
0 0 0 0 0 0 0 0 w w w w w w w w w w w w w w w w w w w w w
chosen(prvih n-1 je u pocetnom stablu odabrano)
0 0 1 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 1 0 0
feature1 feature2 , feature1=broj susjeda, feature2=broj susjeda susjedova
number_11 number_12
number_21 number_22
...
number_n1 number_n2

arhitektura cover filea:
t(broj grafova) puta bude sljedece:
broj_augmenting_edgeova scip_solution_value
edgeovi u stablu pokriveni aue-om 1
edgeovi u stablu pokriveni aue-om 2
edgeovi u stablu pokriveni aue-om 3
.....
edgeovi u stablu pokriveni aue-om broj_augmenting_edgeova

*/

