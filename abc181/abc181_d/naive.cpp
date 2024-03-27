#include<iostream>
using namespace std;
string s;
int cnt[10];
bool dfs(int id,int now)
{
	if(id==3||id==s.size())return now%8==0;
	for(int i=0;i<10;i++)if(cnt[i]>0)
	{
		cnt[i]--;
		if(dfs(id+1,now*10+i))return true;
		cnt[i]++;
	}
	return false;
}
int main()
{
	cin>>s;
	for(int i=0;i<s.size();i++)cnt[s[i]-'0']++;
	cout<<(dfs(0,0)?"Yes":"No")<<endl;
    return 0;
}