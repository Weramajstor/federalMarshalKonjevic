/*
graf se generira da je svaki par bridova spojen s vjerojatnoscu p
*/
void unif( vector< pair<pii,int> > & edges, int n, bool uniform_cost_edges ){//dodaje sve moguce edgevoe i to se onda u mstu itd rjesi
//	srand(time(NULL));
	double p=0.8;//i mali postotak moze ak je n dovoljno velki
	//jer n*n/2 je edgeva i n je vertexa, znaci n/2*p edgeva po vertexu u prosjeku
	
	for(int i=0;i<n;i++){//JEDINI DRUKCIJI DIO ZA SVAKI TESTCASE ova for petlja i sve u njoj > ispod
		for(int j=i+1;j<n;j++){
			
			double x=(double)(rand()%1001)/1000;
			
			if(x<p){
				if(uniform_cost_edges)edges.push_back( pair<pii,int>( pii(i,j) , 1 ) );
				else edges.push_back( pair<pii,int>( pii(i,j) , rand()%1001 ) );
			}
		}
	}
	return;
}


/*
graf stvoren od euklidskih udaljenosti po [1, 10000]x[1, 10000] prostoru
*/
void euc( vector< pair<pii,int> > & edges, int n, bool uniform_cost_edges ){
	/*
		[1,10000]
		f = {2000, 5100, 15000} yield an edge density similar to the one produced by p = {0.1, 0.5, 0.9}
	*/
//	srand(time(NULL));
	double f=4000;
	
	pii tocke[n];
	
	for(int i=0;i<n;i++){
		tocke[i].first=rand()%10000+1;
		tocke[i].second=rand()%10000+1;
	}
	
	for(int i=0;i<n;i++){
		for(int j=i+1;j<n;j++){
			double ix=tocke[i].first-tocke[j].first;
			double iy=tocke[i].second-tocke[j].second;
			
			double d = sqrt( ix*ix + iy*iy );
			
			if( d<f ){
				if(uniform_cost_edges)edges.push_back( pair<pii,int>( pii(i,j) , 1 ) );
				else edges.push_back( pair<pii,int>( pii(i,j) , d ) );
			}
			
		}
	}
	
	return;
}


/*
smallworld benchmark
*/
void smallworld( vector< pair<pii,int> > & edges, int n, bool uniform_cost_edges ){
//	srand(time(NULL));
	double p=0.4;
	
	int d=(int)n*0.3;//sa kolko je susjeda svaki spojen inicijalno
	
	//cout<<d<<endl;
	
	if(d&1)d++;
	d/=2;
	
	//mogu imat dva seta
	map<pii,int> m;
	
	vector< pair<pii,int> > pom;
	
	for(int j=1;j<=d;j++){
		
		for(int i=0;i<n;i++){
			
			int x=i, y=(i+d)%n;
			
			pii par= pii(x, y);
			
			if(uniform_cost_edges)pom.push_back( pair<pii,int>( pii(x,y) , 1 ) );
			else pom.push_back( pair<pii,int>( pii(x,y) , rand()%1001 ) );
			m[pii(x,y)]=m[pii(y,x)]=1;
		}
	}
	
	for( auto e : pom){
		
		double vjer=(double)(rand()%1001)/1000;
		
		if( vjer>p ){
			edges.push_back(e);
			continue;
		}
		
		int x=e.first.first, y=e.first.second, cijena=e.second;
		
		m[pii(x,y)]=m[pii(y,x)]=0;
		
		vector<int> odabir;
		
		for(int g=0;g<n;g++){
			if(g==x || g==y)continue;
			if(!m[pii(x,g)])odabir.push_back(g);
		}
		
		y = odabir[ rand()%odabir.size() ];
		
		edges.push_back( pair<pii,int>( pii(x,y) , cijena ) );	
	}
	
	return;
}


