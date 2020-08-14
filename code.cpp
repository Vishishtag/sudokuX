#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace std;
using namespace __gnu_pbds;
using ll = long long int;
#define INF             1e14
#define dd              double
#define MOD             1000000007
#define my_sizeof(type) ((char *)(&type+1)-(char*)(&type))
#define nl              cout<<endl
#define fill(a,val)     memset(a,val,sizeof(a))
#define mp              make_pair
#define endl            "\n"
#define pb              push_back
#define ff              first
#define ss              second
#define SIZE            200005
#define all(v)          v.begin(),v.end()
#define s(ar,n)         sort(ar,ar+n)
#define rs(ar,n)        sort(ar, ar+n, greater<ll>())
#define oa(a,n)         for(ll i=0;i<n;i++)cout<<a[i]<<" ";nl
#define cn(a,n)         for(ll i=0;i<n;i++)cin>>a[i];
#define maxa(ar,N)      *max_element(ar,ar+N)
#define mina(ar,N)      *min_element(ar,ar+N)
#define fastio()        ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0)
#define printclock      cerr<<"Time : "<<1000*(long double)clock()/(long double)CLOCKS_PER_SEC<<"ms\n";
typedef tree<pair<ll,ll>, null_type, less<pair<ll,ll>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
// find_by_order() - 0 indexing
// order_of_key() - number of elements strictly smaller
ll power(ll x, ll y){
    ll res=1;x=x;
    while(y>0){
        if(y&1)
            res = (res*x);
        y = y>>1;
        x = (x*x);
    }
    return res;
}
ll logtwo(ll n){
    if(n==1)
        return 0;
    return logtwo(n/2)+1;
}
//---------------------------------------GL HF------------------------------------------------------
// -------------------------------------ManavJ07----------------------------------------------------
ll ar[9][9];
ll N=9;
map<pair<ll,ll>,set<ll>> precompute;
bool rowMapNumbers[9][9],columnMapNumbers[9][9];
void takeInput(){
	for(ll i=0;i<N;i++){
    	for(ll j=0;j<N;j++){
    		cin>>ar[i][j];
    	}
    }
}
void preCompute(){
	precompute.clear();
	for(ll i=0;i<N;i++){
    	for(ll j=0;j<N;j++){
    		if(ar[i][j]==0){
    			for(ll k=1;k<=9;k++)
    				precompute[{i,j}].insert(k);
    		}
    		else{
    			rowMapNumbers[i][ar[i][j]-1]=1;
    			columnMapNumbers[j][ar[i][j]-1]=1;
    		}
    	}
    }
    for(auto& box:precompute){
    	ll x=box.ff.ff,y=box.ff.ss;
    	for(ll i=0;i<9;i++){
    		precompute[box.ff].erase(ar[x][i]);
    		precompute[box.ff].erase(ar[i][y]);
    	}
    	ll rowStart=x-x%3;
    	ll colStart=y-y%3;
    	for(ll i=0;i<3;i++){
    		for(ll j=0;j<3;j++){
    			precompute[box.ff].erase(ar[i+rowStart][j+colStart]);
    		}
    	}
    }
}
void printAns(){
	for(ll i=0;i<N;i++){
		for(ll j=0;j<N;j++){
			cout<<ar[i][j]<<" ";
		}
		nl;
	}
}
bool checkRow(){
	for(ll i=1;i<=9;i++){
    	for(ll row=0;row<9;row++){
    		if(rowMapNumbers[row][i-1])
    			continue;
    		ll cnt=0;
    		for(ll col=0;col<9;col++){
    			if(ar[row][col])
    				continue;
    			cnt+=precompute[{row,col}].count(i);
    		}
    		if(cnt>1)
    			continue;
    		for(ll col=0;col<9;col++){
    			if(ar[row][col])
    				continue;
    			if(precompute[{row,col}].count(i))
    				ar[row][col]=i;
    		}
    		return true;
    	}
    }
    return false;
}
bool checkCol(){
	for(ll i=1;i<=9;i++){
    	for(ll col=0;col<9;col++){
    		if(columnMapNumbers[col][i-1])
    			continue;
    		ll cnt=0;
    		for(ll row=0;row<9;row++){
    			if(ar[row][col])
    				continue;
    			cnt+=precompute[{row,col}].count(i);
    		}
    		if(cnt>1)
    			continue;
    		for(ll row=0;row<9;row++){
    			if(ar[row][col])
    				continue;
    			if(precompute[{row,col}].count(i))
    				ar[row][col]=i;
    		}
    		return true;
    	}
    }
    return false;
}
bool checkBox(){
	for(ll i=1;i<=9;i++){
		for(ll startRow=0;startRow<9;startRow+=3){
			for(ll startCol=0;startCol<9;startCol+=3){
				bool exists=0;
				for(ll x=0;x<3;x++){
					for(ll y=0;y<3;y++){
						exists|=(ar[x+startRow][y+startCol]==i);
					}
				}
				if(exists)
					continue;
				ll cnt=0;
				for(ll x=0;x<3;x++){
					for(ll y=0;y<3;y++){
						if(ar[x+startRow][y+startCol])
							continue;
						cnt+=precompute[{x+startRow,y+startCol}].count(i);
					}
				}
				if(cnt!=1)
					continue;
				for(ll x=0;x<3;x++){
					for(ll y=0;y<3;y++){
						if(ar[x+startRow][y+startCol])
							continue;
						if(precompute[{x+startRow,y+startCol}].count(i))
							ar[x+startRow][startCol+y]=i;
					}
				}
				return true;
			}
		}
	}
	return false;
}
bool FindUnassignedLocation(int& row, int& col);
bool isSafe(int row,int col, int num);
bool SolveSudoku()
{
    int row, col;
    if (!FindUnassignedLocation(row, col))
        return true;
    for (int num = 1; num <= 9; num++) {
        if (isSafe(row, col, num)) {
            ar[row][col] = num;
            if (SolveSudoku())
                return true;
            ar[row][col] = 0;
        }
    }
    return false;
}
bool FindUnassignedLocation(int& row, int& col)
{
    for (row = 0; row < N; row++)
        for (col = 0; col < N; col++)
            if (ar[row][col] == 0)
                return true;
    return false;
}
bool UsedInRow(int row, int num)
{
    for (int col = 0; col < N; col++)
        if (ar[row][col] == num)
            return true;
    return false;
}
bool UsedInCol(int col, int num)
{
    for (int row = 0; row < N; row++)
        if (ar[row][col] == num)
            return true;
    return false;
}
bool UsedInBox(int boxStartRow,int boxStartCol, int num)
{
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            if (ar[row + boxStartRow][col + boxStartCol]== num)
                return true;
    return false;
}
bool isSafe(int row,int col, int num)
{
    return !UsedInRow(row, num) && !UsedInCol(col, num) && !UsedInBox(row - row % 3,col - col % 3, num) && ar[row][col] == 0;
}
signed main()
{
	fastio();
    #ifndef ONLINE_JUDGE
    freopen("input.txt","r",stdin);
    freopen("output.txt","w",stdout);
    #endif
    takeInput();
    preCompute();
    while(precompute.size()){
    	pair<ll,ll> f={-1,-1};
    	for(auto& box:precompute){
    		if(box.ss.size()==1){
    			f=box.ff;
    			break;
    		}
    	}
    	if(f.ff!=-1){
    		ar[f.ff][f.ss]=*precompute[f].begin();
    		preCompute();
    		continue;
    	}
    	if(checkRow()){
    		preCompute();
    		continue;
    	}
    	if(checkCol()){
    		preCompute();
    		continue;
    	}
    	if(checkBox()){
    		preCompute();
    		continue;
    	}
    	SolveSudoku();
    	preCompute();
    }
    printAns();
}
