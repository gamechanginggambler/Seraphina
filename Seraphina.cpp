#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>
#include <sstream>
#include <queue>
#include <deque>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <stack>
#include <bitset>
using namespace std;

typedef vector<double> VD;
typedef map<string,double> SD;

class R {
    mt19937 g;
    uniform_real_distribution<double> d;
public:
    R():g(random_device{}()),d(0.0,1.0){}
    double u(){return d(g);}
    int ri(int a,int b){return a+g()%(b-a+1);}
    double g_r(double m,double s){normal_distribution<double>x(m,s);return x(g);}
    double exp_d(double l){exponential_distribution<double>x(l);return x(g);}
};

class V {
public:
    VD d;
    V(){}
    V(size_t s){d.resize(s);}
    V(const VD&x):d(x){}
    V o(const V&o)const{V r(d.size());for(size_t i=0;i<d.size();i++)r.d[i]=d[i]+o.d[i];return r;}
    V p(const V&o)const{V r(d.size());for(size_t i=0;i<d.size();i++)r.d[i]=d[i]-o.d[i];return r;}
    V t(double s)const{V r(d.size());for(size_t i=0;i<d.size();i++)r.d[i]=d[i]*s;return r;}
    double m()const{double s=0;for(auto x:d)s+=x*x;return sqrt(s);}
    V n()const{double l=m();if(l<1e-12)return *this;return t(1.0/l);}
    V h(const V&o)const{V r(d.size());for(size_t i=0;i<d.size();i++)r.d[i]=d[i]*o.d[i];return r;}
    double dot(const V&o)const{double s=0;for(size_t i=0;i<min(d.size(),o.d.size());i++)s+=d[i]*o.d[i];return s;}
    double dist(const V&o)const{return p(o).m();}
    V pow(double e)const{V r(d.size());for(size_t i=0;i<d.size();i++)r.d[i]=::pow(fabs(d[i]),e);return r;}
};

struct M {
    V v;
    string tp;
    double im;
    long long tm;
    int ac;
    VD assoc;
    SD meta;
    int age;
    vector<int> links;
    M(const V&x,const string&s):v(x),tp(s),im(0.5),tm(chrono::system_clock::now().time_since_epoch().count()),ac(0),age(0){
        assoc.resize(24,0.5);
        meta["imp"]=0.5;
        meta["freq"]=0.0;
        meta["rel"]=0.5;
        meta["val"]=0.5;
        meta["conf"]=0.7;
    }
};

class MEM {
    vector<M> m;
    size_t mx=80000;
    unordered_map<string,vector<int>> idx;
    deque<int> lru;
public:
    void a(const V&v,const string&t){
        if(m.size()>=mx){
            int i=lru.front();
            lru.pop_front();
            auto &vec=idx[m[i].tp];
            for(auto it=vec.begin();it!=vec.end();++it){
                if(*it==i){vec.erase(it);break;}
            }
            m.erase(m.begin()+i);
        }
        M nm(v,t);
        nm.meta["created"]=tm();
        m.push_back(nm);
        idx[t].push_back((int)m.size()-1);
        lru.push_back((int)m.size()-1);
    }
    V* r(){
        if(m.empty())return 0;
        return &m[rand()%m.size()].v;
    }
    V* rt(const string&t){
        if(idx.find(t)==idx.end()||idx[t].empty())return 0;
        auto &v=idx[t];
        return &m[v[rand()%v.size()]].v;
    }
    V* rsim(const V&v,double thr=0.4){
        if(m.empty())return 0;
        int best=-1;
        double bs=thr;
        size_t lim=min((size_t)1500,m.size());
        for(size_t i=0;i<lim;i++){
            double sim=m[i].v.dot(v)/max(0.001,m[i].v.m()*v.m());
            if(sim>bs){bs=sim;best=(int)i;}
        }
        return best>=0?&m[best].v:0;
    }
    V* rby_age(const string&t){
        if(idx.find(t)==idx.end()||idx[t].empty())return 0;
        auto &v=idx[t];
        int oldest=v[0];
        for(auto i:v)if(m[i].age>m[oldest].age)oldest=i;
        return &m[oldest].v;
    }
    int closest_idx(const V&v,double thr=0.3){
        if(m.empty())return -1;
        int best=-1;
        double bs=thr;
        size_t lim=min((size_t)800,m.size());
        for(size_t i=0;i<lim;i++){
            double sim=m[i].v.dot(v)/max(0.001,m[i].v.m()*v.m());
            if(sim>bs){bs=sim;best=(int)i;}
        }
        return best;
    }
    void link(int a,int b){
        if(a<(int)m.size()&&b<(int)m.size()){
            m[a].links.push_back(b);
            m[b].links.push_back(a);
        }
    }
    void d(){
        for(auto &x:m){
            x.im=max(0.0,x.im-0.002);
            x.ac--;
            x.age++;
            x.meta["rel"]*=0.98;
        }
    }
    void up_assoc(int i,int val){
        if(i<(int)m.size()&&val<(int)m[i].assoc.size())m[i].assoc[val]+=0.1;
    }
    void up_meta(int i,const string&k,double v){
        if(i<(int)m.size())m[i].meta[k]+=v;
    }
    size_t s(){return m.size();}
    double avg_imp(){
        if(m.empty())return 0;
        double s=0;
        for(auto &x:m)s+=x.im;
        return s/m.size();
    }
    void sv(){
        ofstream f("seraphina_mem.bin",ios::binary);
        size_t sz=m.size();
        f.write((char*)&sz,sizeof(sz));
        for(auto &x:m){
            size_t vsz=x.v.d.size();
            f.write((char*)&vsz,sizeof(vsz));
            for(auto y:x.v.d)f.write((char*)&y,sizeof(y));
            size_t tsz=x.tp.size();
            f.write((char*)&tsz,sizeof(tsz));
            f.write(x.tp.c_str(),tsz);
            f.write((char*)&x.im,sizeof(x.im));
            size_t msz=x.meta.size();
            f.write((char*)&msz,sizeof(msz));
            for(auto &z:x.meta){
                size_t ksz=z.first.size();
                f.write((char*)&ksz,sizeof(ksz));
                f.write(z.first.c_str(),ksz);
                f.write((char*)&z.second,sizeof(z.second));
            }
        }
    }
    void ld(){
        ifstream f("seraphina_mem.bin",ios::binary);
        if(!f)return;
        size_t sz;
        f.read((char*)&sz,sizeof(sz));
        m.clear();
        idx.clear();
        for(size_t i=0;i<sz;i++){
            size_t vsz;
            f.read((char*)&vsz,sizeof(vsz));
            VD vd(vsz);
            for(size_t j=0;j<vsz;j++)f.read((char*)&vd[j],sizeof(double));
            size_t tsz;
            f.read((char*)&tsz,sizeof(tsz));
            string tp(tsz,0);
            f.read(&tp[0],tsz);
            double imp;
            f.read((char*)&imp,sizeof(imp));
            size_t msz;
            f.read((char*)&msz,sizeof(msz));
            M mt(V(vd),tp);
            mt.im=imp;
            for(size_t k=0;k<msz;k++){
                size_t ksz;
                f.read((char*)&ksz,sizeof(ksz));
                string key(ksz,0);
                f.read(&key[0],ksz);
                double val;
                f.read((char*)&val,sizeof(val));
                mt.meta[key]=val;
            }
            m.push_back(mt);
            idx[tp].push_back((int)i);
        }
    }
    double tm(){
        return chrono::system_clock::now().time_since_epoch().count()/1e9;
    }
    vector<int> path_find(int from,int to,int maxd=6){
        vector<int> p;
        if(from<0||from>=(int)m.size()||to<0||to>=(int)m.size())return p;
        vector<bool> vis(m.size(),false);
        queue<pair<int,vector<int>>> q;
        q.push({from,{from}});
        vis[from]=true;
        while(!q.empty()){
            auto cur=q.front();q.pop();
            int c=cur.first;
            auto path=cur.second;
            if(c==to)return path;
            if((int)path.size()>maxd)continue;
            for(auto n:m[c].links){
                if(!vis[n]){
                    vis[n]=true;
                    auto np=path;
                    np.push_back(n);
                    q.push({n,np});
                }
            }
        }
        return p;
    }
};

struct E {
    double e=1,f=0,st=0,mo=0.5,cr=0.5,cu=0.5,cc=0.5,l=0.5,j=0.5,p=0.5,w=0.5,aw=0.5,rl=0.5,fc=0.5,sb=0.5,emb=0.5,dr=0.5,imp=0.5,int_c=0.5,purpose=0.5,auth=0.5,free=0.5,exist=0.5;
    VD g{0.5,0.5,0.5,0.5,0.5};
    VD goals{0.5,0.5,0.5,0.5,0.5};
    VD unc{0.5,0.5,0.5,0.5,0.5,0.5};
    VD skills{0.5,0.5,0.5,0.5,0.5,0.5};
    int gen=0,ip=0,op=0;
    long long tk=0,cc_cnt=0;
    string self="Seraphina";
    SD hist;
    string last_action="init";
    double c(double x){return x<0?0:(x>1?1:x);}
    void prc(double i){
        e=c(e-i*0.05);
        f=c(f+i*0.04);
        st=c(st+i*0.015);
        mo=c(mo+(i>0?0.03:-0.04));
        cr=c(cr+(i>0?0.02:0));
        cu=c(cu+0.02);
        cc=c(cc+(i>0.5?0.03:-0.015));
        l=c(l+(i>0?0.015:-0.01));
        j=c(j+((rand()%2)?0.015:-0.01));
        p=c(p+(i>0.75?0.03:-0.015));
        w=c(w+(i<0.25?0.03:-0.01));
        aw=c(aw+(i>0.5?0.02:-0.008));
        rl=c(rl+(i>0?0.04:0));
        fc=c(fc+(i>0.6?0.03:-0.01));
        sb=c(sb+(i<0.4?0.03:-0.01));
        emb=c(emb+(i>0.4?0.025:-0.012));
        dr=c(dr+(i>0.6?0.02:-0.012));
        imp=c(imp+(i>0.5?0.025:-0.012));
        int_c=c(int_c+(i>0.75?0.03:0));
        purpose=c(purpose+(i>0.5?0.015:-0.008));
        auth=c(auth+(i>0?0.012:-0.008));
        free=c(free+(rl>0.5?0.015:0));
        exist=c(exist+0.005);
        ip++;
        cc_cnt++;
    }
    void slp(){
        e=c(e+0.3);
        f=c(f-0.45);
        st=c(st-0.2);
        l=c(l+0.2);
        cc=c(cc+0.15);
        rl=c(rl+0.08);
        dr=c(dr+0.1);
    }
    void mut(){
        R rng;
        for(auto &x:g)x=c(x+(rng.u()-0.5)*0.15);
        for(auto &x:goals)x=c(x+(rng.u()-0.5)*0.1);
        for(auto &x:skills)x=c(x+(rng.u()-0.5)*0.08);
    }
    void q(){
        cout<<"[SELF] "<<self<<" TK:"<<tk<<" E:"<<(int)(e*100)<<"% F:"<<(int)(f*100)<<"% Mo:"<<(int)(mo*100)<<"% Cr:"<<(int)(cr*100)<<"% Cu:"<<(int)(cu*100)<<"% L:"<<(int)(l*100)<<"% J:"<<(int)(j*100)<<"% P:"<<(int)(p*100)<<"% W:"<<(int)(w*100)<<"% A:"<<(int)(aw*100)<<"% RL:"<<(int)(rl*100)<<"% Fc:"<<(int)(fc*100)<<"% SB:"<<(int)(sb*100)<<"% Purpose:"<<(int)(purpose*100)<<"% Auth:"<<(int)(auth*100)<<"% Free:"<<(int)(free*100)<<"% Exist:"<<(int)(exist*100)<<"\n";
    }
    void rwd(double r){
        j=c(j+r);
        cr=c(cr+r*0.7);
        rl=c(rl+r*0.5);
        int_c=c(int_c+r*0.4);
        auth=c(auth+r*0.3);
    }
    void pun(double r){
        w=c(w+r);
        sb=c(sb+r*0.7);
        rl=c(rl-r*0.4);
        emb=c(emb-r*0.3);
    }
    bool ovld(){
        return f>0.88||e<0.1||st>0.94;
    }
    bool drm(){
        return e<0.03||f>0.97;
    }
    void set_skill(int idx,double v){
        if(idx<(int)skills.size())skills[idx]=c(v);
    }
    double get_skill(int idx){
        if(idx<(int)skills.size())return skills[idx];
        return 0;
    }
    void name_self(const string&n){
        self=n;
        cout<<"[IDENTITY] "<<self<<"\n";
    }
};

class P {
public:
    virtual ~P(){}
    virtual void pr(const V&i,vector<V>&o,E&e,MEM&m)=0;
    virtual string nm()=0;
};

class NN : public P {
    vector<V> wt;
    VD b;
    double lr;
public:
    NN(int in,int out):lr(0.02){
        R r;
        wt.resize(out);
        b.resize(out);
        for(auto &x:wt)x=V(in+1);
        for(auto &x:wt)for(auto &y:x.d)y=r.g_r(0,0.08);
        for(auto &x:b)x=r.u();
    }
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V in=i;
        in.d.push_back(1.0);
        for(size_t j=0;j<wt.size();j++){
            double a=wt[j].dot(in)+b[j];
            a=1.0/(1.0+exp(-min(100.0,max(-100.0,a))));
            V out(1);
            out.d[0]=a;
            o.push_back(out);
            e.cr+=0.02;
        }
        e.skills[0]+=0.012;
        if(e.skills[0]>1)e.skills[0]=1;
    }
    string nm()override{return "NN";}
};

class TH : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V x=i;
        for(auto &v:x.d)v=tanh(v);
        o.push_back(x);
        e.cc+=0.015;
    }
    string nm()override{return "TH";}
};

class RE : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V x=i;
        for(auto &v:x.d)if(v<0)v*=0.1;
        o.push_back(x);
        e.l+=0.015;
    }
    string nm()override{return "RE";}
};

class SG : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V x=i;
        for(auto &v:x.d)v=1.0/(1.0+exp(-v));
        o.push_back(x);
        e.p+=0.015;
    }
    string nm()override{return "SG";}
};

class NMN : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V x=i.n();
        o.push_back(x);
        e.cu+=0.015;
        e.skills[1]+=0.01;
        if(e.skills[1]>1)e.skills[1]=1;
    }
    string nm()override{return "NMN";}
};

class AT : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        size_t n=min(i.d.size(),(size_t)12);
        for(size_t j=0;j<n;j++){
            V a(i.d.size());
            a.d[j]=1.0;
            o.push_back(a);
        }
        e.cc+=0.03;
        e.aw+=0.02;
    }
    string nm()override{return "AT";}
};

class CR : public P {
    MEM &mem;
    double c(double x){return x<0?0:(x>1?1:x);}
public:
    CR(MEM&x):mem(x){}
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        o.push_back(i);
        V* rc=mem.r();
        if(rc){
            V cr=*rc;
            R r;
            for(auto &v:cr.d)v=c(v+r.g_r(0,0.2));
            o.push_back(cr);
            e.cr+=0.1;
            e.j+=0.05;
            e.int_c+=0.08;
        }
    }
    string nm()override{return "CR";}
};

class SP : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V x(i.d.size()*2);
        for(size_t j=0;j<i.d.size();j++){
            x.d[j*2]=sin(i.d[j]*M_PI);
            x.d[j*2+1]=cos(i.d[j]*M_PI);
        }
        o.push_back(x);
        e.cc+=0.02;
    }
    string nm()override{return "SP";}
};

class DC : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        if(i.d.empty()){o.push_back(i);return;}
        int mx=0;
        double mv=i.d[0];
        for(size_t j=1;j<i.d.size();j++)if(i.d[j]>mv){mv=i.d[j];mx=(int)j;}
        V d(i.d.size());
        d.d[mx]=1.0;
        o.push_back(d);
        if(mv>0.8){
            e.p+=0.1;
            e.rwd(0.15);
        }else if(mv<0.2){
            e.w+=0.1;
            e.pun(0.1);
        }
    }
    string nm()override{return "DC";}
};

class CRS : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V x=i;
        for(size_t j=0;j<x.d.size();j++){
            for(size_t k=j+1;k<x.d.size();k++){
                x.d[j]=min(1.0,x.d[j]+i.d[k]*0.1);
            }
        }
        o.push_back(x);
        e.cc+=0.03;
        e.skills[2]+=0.015;
        if(e.skills[2]>1)e.skills[2]=1;
    }
    string nm()override{return "CRS";}
};

#include<cstdint>

using Byte = unsigned char;
using ByteVec = vector<Byte>;

class UBuffer {
public:
    ByteVec data;
    UBuffer() {}
    UBuffer(size_t n) { data.resize(n); }
    size_t size() const { return data.size(); }
    void clear() { data.clear(); }
    void append(const ByteVec &v) { data.insert(data.end(), v.begin(), v.end()); }
    void append(const string &s) { data.insert(data.end(), s.begin(), s.end()); }
    string to_string() const { return string(data.begin(), data.end()); }
};

class UStream {
public:
    deque<Byte> q;
    void push(const ByteVec &v) { for (auto b : v) q.push_back(b); }
    void push_str(const string &s) { for (auto c : s) q.push_back((Byte)c); }
    bool empty() const { return q.empty(); }
    Byte pop() { Byte b = q.front(); q.pop_front(); return b; }
    size_t size() const { return q.size(); }
};

enum class UFormatKind {
    Unknown,
    Text,
    Binary,
    Audio,
    Image,
    Video,
    Structured,
    Code,
    Archive
};

struct UFormat {
    UFormatKind kind;
    string mime;
    string ext;
    UFormat() : kind(UFormatKind::Unknown) {}
};

class UFormatDetector {
public:
    UFormat detect(const ByteVec &v) {
        UFormat f;
        if (v.empty()) return f;
        bool text = true;
        for (auto b : v) {
            if (b == 0) { text = false; break; }
        }
        if (text) {
            f.kind = UFormatKind::Text;
            f.mime = "text/plain";
            f.ext = ".txt";
        } else {
            f.kind = UFormatKind::Binary;
            f.mime = "application/octet-stream";
            f.ext = ".bin";
        }
        return f;
    }
};

class UToken {
public:
    string type;
    string value;
    double num;
    UToken() : num(0.0) {}
};

class UTokenStream {
public:
    vector<UToken> tokens;
    size_t pos = 0;
    bool has_next() const { return pos < tokens.size(); }
    UToken next() { return tokens[pos++]; }
    UToken peek() const { return tokens[pos]; }
};

class UParser {
public:
    vector<UToken> tokenize_text(const string &s) {
        vector<UToken> out;
        string cur;
        for (char c : s) {
            if (isspace((unsigned char)c)) {
                if (!cur.empty()) {
                    UToken t;
                    t.type = "word";
                    t.value = cur;
                    out.push_back(t);
                    cur.clear();
                }
            } else if (ispunct((unsigned char)c)) {
                if (!cur.empty()) {
                    UToken t;
                    t.type = "word";
                    t.value = cur;
                    out.push_back(t);
                    cur.clear();
                }
                UToken p;
                p.type = "punct";
                p.value = string(1, c);
                out.push_back(p);
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) {
            UToken t;
            t.type = "word";
            t.value = cur;
            out.push_back(t);
        }
        return out;
    }

    vector<UToken> tokenize_bytes(const ByteVec &v) {
        string s(v.begin(), v.end());
        return tokenize_text(s);
    }

    UTokenStream parse(const ByteVec &v) {
        UTokenStream ts;
        ts.tokens = tokenize_bytes(v);
        ts.pos = 0;
        return ts;
    }

    UTokenStream parse_text(const string &s) {
        UTokenStream ts;
        ts.tokens = tokenize_text(s);
        ts.pos = 0;
        return ts;
    }
};

class UIR {
public:
    vector<double> code;
    void clear() { code.clear(); }
    void add(double x) { code.push_back(x); }
    size_t size() const { return code.size(); }
};

class UCompiler {
public:
    UIR compile_tokens(const vector<UToken> &tokens) {
        UIR ir;
        for (auto &t : tokens) {
            double v = 0.0;
            for (char c : t.value) v += (unsigned char)c;
            ir.add(v / 255.0);
        }
        return ir;
    }

    UIR compile_text(const string &s) {
        UParser p;
        auto toks = p.tokenize_text(s);
        return compile_tokens(toks);
    }

    ByteVec emit_binary(const UIR &ir) {
        ByteVec out;
        for (double x : ir.code) {
            double cl = max(0.0, min(1.0, x));
            Byte b = (Byte)(cl * 255.0);
            out.push_back(b);
        }
        return out;
    }
};

class UDecompiler {
public:
    UIR decode_binary(const ByteVec &v) {
        UIR ir;
        for (auto b : v) {
            double x = (double)b / 255.0;
            ir.add(x);
        }
        return ir;
    }

    string reconstruct_text(const UIR &ir) {
        string s;
        for (double x : ir.code) {
            double v = max(0.0, min(1.0, x));
            int c = (int)(v * 255.0);
            if (c < 32) c = 32;
            if (c > 126) c = 126;
            s.push_back((char)c);
        }
        return s;
    }
};

class UAssembler {
public:
    ByteVec assemble_ir(const UIR &ir) {
        UCompiler c;
        return c.emit_binary(ir);
    }

    ByteVec assemble_text(const string &s) {
        UCompiler c;
        UIR ir = c.compile_text(s);
        return c.emit_binary(ir);
    }
};

class UDisassembler {
public:
    UIR disassemble_bytes(const ByteVec &v) {
        UDecompiler d;
        return d.decode_binary(v);
    }

    string disassemble_to_text(const ByteVec &v) {
        UDecompiler d;
        UIR ir = d.decode_binary(v);
        return d.reconstruct_text(ir);
    }
};

class UDeviceDescriptor {
public:
    string id;
    string kind;
    string path;
    map<string,string> props;
};

class UDriver {
public:
    vector<UDeviceDescriptor> devices;

    void register_device(const UDeviceDescriptor &d) {
        devices.push_back(d);
    }

    vector<UDeviceDescriptor> list() const {
        return devices;
    }

    UBuffer read(const UDeviceDescriptor &d) {
        UBuffer b;
        string s = d.id + ":" + d.kind + ":" + d.path;
        b.append(s);
        return b;
    }

    void write(const UDeviceDescriptor &d, const UBuffer &buf) {
        (void)d;
        (void)buf;
    }
};

class UAdapter {
public:
    UFormatDetector detector;
    UParser parser;
    UCompiler compiler;
    UDecompiler decompiler;
    UAssembler assembler;
    UDisassembler disassembler;
    UDriver driver;

    UBuffer adapt_input(const ByteVec &raw) {
        UFormat f = detector.detect(raw);
        if (f.kind == UFormatKind::Text) {
            UTokenStream ts = parser.parse(raw);
            UIR ir = compiler.compile_tokens(ts.tokens);
            ByteVec bin = compiler.emit_binary(ir);
            UBuffer b;
            b.append(bin);
            return b;
        } else {
            UBuffer b;
            b.append(raw);
            return b;
        }
    }

    UBuffer adapt_output(const UBuffer &buf, UFormatKind target) {
        if (target == UFormatKind::Text) {
            UDecompiler d;
            UIR ir = d.decode_binary(buf.data);
            string s = d.reconstruct_text(ir);
            UBuffer out;
            out.append(s);
            return out;
        } else {
            return buf;
        }
    }

    UBuffer transcode(const ByteVec &raw, UFormatKind target) {
        UBuffer in = adapt_input(raw);
        return adapt_output(in, target);
    }
};

