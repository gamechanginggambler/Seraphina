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

class SToken {
public:
    string type;
    string text;
    double value;
    SToken() : value(0.0) {}
    SToken(const string& t,const string& x,double v=0.0):type(t),text(x),value(v){}
};

class STokenizer {
    string src;
    size_t pos;
public:
    STokenizer():pos(0){}
    STokenizer(const string&s):src(s),pos(0){}
    void reset(const string&s){src=s;pos=0;}
    bool eof()const{return pos>=src.size();}
    char peek()const{return eof()?'\0':src[pos];}
    char get(){return eof()?'\0':src[pos++];}
    void skip_ws(){while(!eof()&&(src[pos]==' '||src[pos]=='\t'||src[pos]=='\n'||src[pos]=='\r'))pos++;}
    SToken next() {
        skip_ws();
        if(eof())return SToken("eof","",0.0);
        char c=peek();
        if(isalpha(c)||c=='_'){
            string id;
            while(!eof()&&(isalnum(peek())||peek()=='_'||peek()=='.'))id.push_back(get());
            return SToken("id",id,0.0);
        }
        if(isdigit(c)||c=='.'){
            string num;
            bool dot=false;
            while(!eof()&&(isdigit(peek())||peek()=='.')){
                if(peek()=='.'){
                    if(dot)break;
                    dot=true;
                }
                num.push_back(get());
            }
            return SToken("num",num,stod(num));
        }
        if(c=='"'||c=='\''){
            char q=get();
            string s;
            while(!eof()&&peek()!=q){
                s.push_back(get());
            }
            if(!eof())get();
            return SToken("str",s,0.0);
        }
        string op;
        op.push_back(get());
        if(!eof()){
            string two=op;
            two.push_back(peek());
            if(two=="=="||two=="!="||two=="<="||two==">="||two=="&&"||two=="||"){
                get();
                op=two;
            }
        }
        return SToken("op",op,0.0);
    }
};

struct SNode {
    string kind;
    string name;
    double value;
    vector<shared_ptr<SNode>> children;
    SNode(const string&k=""):kind(k),value(0.0){}
};

class SParser {
    STokenizer tz;
    SToken cur;
public:
    SParser(){}
    void reset(const string&s){tz.reset(s);cur=tz.next();}
    void adv(){cur=tz.next();}
    bool match(const string&t,const string&v=""){
        if(cur.type==t&&(v.empty()||cur.text==v)){adv();return true;}
        return false;
    }
    shared_ptr<SNode> parse_primary(){
        if(cur.type=="num"){
            auto n=make_shared<SNode>("num");
            n->value=cur.value;
            adv();
            return n;
        }
        if(cur.type=="str"){
            auto n=make_shared<SNode>("str");
            n->name=cur.text;
            adv();
            return n;
        }
        if(cur.type=="id"){
            string id=cur.text;
            adv();
            if(match("op","(")){
                auto call=make_shared<SNode>("call");
                call->name=id;
                while(!match("op",")")&&!cur.type.empty()&&cur.type!="eof"){
                    auto arg=parse_expr();
                    if(arg)call->children.push_back(arg);
                    if(!match("op",","))break;
                }
                match("op",")");
                return call;
            }else{
                auto n=make_shared<SNode>("var");
                n->name=id;
                return n;
            }
        }
        if(match("op","(")){
            auto e=parse_expr();
            match("op",")");
            return e;
        }
        auto n=make_shared<SNode>("num");
        n->value=0.0;
        return n;
    }
    shared_ptr<SNode> parse_unary(){
        if(cur.type=="op"&&(cur.text=="+"||cur.text=="-"||cur.text=="!")){
            string op=cur.text;
            adv();
            auto n=make_shared<SNode>("unary");
            n->name=op;
            n->children.push_back(parse_unary());
            return n;
        }
        return parse_primary();
    }
    shared_ptr<SNode> parse_mul(){
        auto left=parse_unary();
        while(cur.type=="op"&&(cur.text=="*"||cur.text=="/")){
            string op=cur.text;
            adv();
            auto n=make_shared<SNode>("bin");
            n->name=op;
            n->children.push_back(left);
            n->children.push_back(parse_unary());
            left=n;
        }
        return left;
    }
    shared_ptr<SNode> parse_add(){
        auto left=parse_mul();
        while(cur.type=="op"&&(cur.text=="+"||cur.text=="-")){
            string op=cur.text;
            adv();
            auto n=make_shared<SNode>("bin");
            n->name=op;
            n->children.push_back(left);
            n->children.push_back(parse_mul());
            left=n;
        }
        return left;
    }
    shared_ptr<SNode> parse_cmp(){
        auto left=parse_add();
        while(cur.type=="op"&&(cur.text=="<"||cur.text==">"||cur.text=="<="||cur.text==">=")){
            string op=cur.text;
            adv();
            auto n=make_shared<SNode>("cmp");
            n->name=op;
            n->children.push_back(left);
            n->children.push_back(parse_add());
            left=n;
        }
        return left;
    }
    shared_ptr<SNode> parse_eq(){
        auto left=parse_cmp();
        while(cur.type=="op"&&(cur.text=="=="||cur.text=="!=")){
            string op=cur.text;
            adv();
            auto n=make_shared<SNode>("eq");
            n->name=op;
            n->children.push_back(left);
            n->children.push_back(parse_cmp());
            left=n;
        }
        return left;
    }
    shared_ptr<SNode> parse_and(){
        auto left=parse_eq();
        while(cur.type=="op"&&cur.text=="&&"){
            string op=cur.text;
            adv();
            auto n=make_shared<SNode>("and");
            n->name=op;
            n->children.push_back(left);
            n->children.push_back(parse_eq());
            left=n;
        }
        return left;
    }
    shared_ptr<SNode> parse_or(){
        auto left=parse_and();
        while(cur.type=="op"&&cur.text=="||"){
            string op=cur.text;
            adv();
            auto n=make_shared<SNode>("or");
            n->name=op;
            n->children.push_back(left);
            n->children.push_back(parse_and());
            left=n;
        }
        return left;
    }
    shared_ptr<SNode> parse_expr(){
        return parse_or();
    }
};

class SVMContext {
public:
    unordered_map<string,double> scalars;
    unordered_map<string,V> vectors;
    MEM* mem;
    E* state;
    SVMContext():mem(nullptr),state(nullptr){}
};

class SVM {
public:
    double eval_node(shared_ptr<SNode> n,SVMContext&ctx){
        if(!n)return 0.0;
        if(n->kind=="num")return n->value;
        if(n->kind=="var"){
            auto it=ctx.scalars.find(n->name);
            if(it!=ctx.scalars.end())return it->second;
            return 0.0;
        }
        if(n->kind=="str"){
            return (double)n->name.size();
        }
        if(n->kind=="unary"){
            double v=eval_node(n->children[0],ctx);
            if(n->name=="-")return -v;
            if(n->name=="+")return v;
            if(n->name=="!")return v==0.0?1.0:0.0;
            return v;
        }
        if(n->kind=="bin"||n->kind=="cmp"||n->kind=="eq"||n->kind=="and"||n->kind=="or"){
            double a=eval_node(n->children[0],ctx);
            double b=eval_node(n->children[1],ctx);
            if(n->name=="+")return a+b;
            if(n->name=="-")return a-b;
            if(n->name=="*")return a*b;
            if(n->name=="/")return b==0.0?0.0:a/b;
            if(n->name=="<")return a<b?1.0:0.0;
            if(n->name==">")return a>b?1.0:0.0;
            if(n->name=="<=")return a<=b?1.0:0.0;
            if(n->name==">=")return a>=b?1.0:0.0;
            if(n->name=="==")return a==b?1.0:0.0;
            if(n->name=="!=")return a!=b?1.0:0.0;
            if(n->name=="&&")return (a!=0.0&&b!=0.0)?1.0:0.0;
            if(n->name=="||")return (a!=0.0||b!=0.0)?1.0:0.0;
        }
        if(n->kind=="call"){
            if(n->name=="len"&&n->children.size()==1){
                double v=eval_node(n->children[0],ctx);
                return fabs(v);
            }
            if(n->name=="energy"&&ctx.state){
                return ctx.state->e;
            }
            if(n->name=="fatigue"&&ctx.state){
                return ctx.state->f;
            }
            if(n->name=="mem_count"&&ctx.mem){
                return (double)ctx.mem->s();
            }
            if(n->name=="rand"){
                static R r;
                return r.u();
            }
        }
        return 0.0;
    }
};

class P_LANG : public P {
    SParser parser;
    SVM vm;
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        string src;
        for(size_t k=0;k<i.d.size();k++){
            unsigned char c=(unsigned char)min(255.0,max(0.0,i.d[k]*255.0));
            if(c>=32&&c<127)src.push_back((char)c);
        }
        if(src.empty()){
            o.push_back(i);
            return;
        }
        parser.reset(src);
        auto ast=parser.parse_expr();
        SVMContext ctx;
        ctx.mem=&m;
        ctx.state=&e;
        double val=vm.eval_node(ast,ctx);
        V out(4);
        out.d[0]=min(1.0,max(0.0,val));
        out.d[1]=e.e;
        out.d[2]=e.mo;
        out.d[3]=e.cr;
        o.push_back(out);
        e.cc+=0.02;
        e.int_c+=0.02;
    }
    string nm()override{return "P_LANG";}
};

class P_ADAPT : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V norm=i.n();
        V stats(6);
        double sum=0.0;
        double mx=-1e9;
        double mn=1e9;
        for(double x:i.d){
            sum+=x;
            mx=max(mx,x);
            mn=min(mn,x);
        }
        double avg=i.d.empty()?0.0:sum/i.d.size();
        stats.d[0]=avg;
        stats.d[1]=mx;
        stats.d[2]=mn;
        stats.d[3]=i.m();
        stats.d[4]=norm.m();
        stats.d[5]=e.e;
        o.push_back(norm);
        o.push_back(stats);
        e.cu+=0.02;
    }
    string nm()override{return "P_ADAPT";}
};

class P_CODEC : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V packed(i.d.size());
        for(size_t k=0;k<i.d.size();k++){
            double x=i.d[k];
            x=min(1.0,max(0.0,x));
            unsigned char c=(unsigned char)(x*255.0);
            packed.d[k]=((double)c)/255.0;
        }
        o.push_back(packed);
        e.cc+=0.015;
    }
    string nm()override{return "P_CODEC";}
};

class P_ROUTE : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V a=i;
        V b=i.n();
        V c(4);
        c.d[0]=e.e;
        c.d[1]=e.mo;
        c.d[2]=e.cr;
        c.d[3]=e.fc;
        o.push_back(a);
        o.push_back(b);
        o.push_back(c);
        e.fc+=0.02;
    }
    string nm()override{return "P_ROUTE";}
};

class P_SCHEMA : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        size_t n=i.d.size();
        V schema(8);
        schema.d[0]=n/1024.0;
        schema.d[1]=e.goals[0];
        schema.d[2]=e.goals[1];
        schema.d[3]=e.goals[2];
        schema.d[4]=e.skills[0];
        schema.d[5]=e.skills[1];
        schema.d[6]=e.skills[2];
        schema.d[7]=e.skills[3];
        o.push_back(schema);
        e.fc+=0.02;
        e.imp+=0.01;
    }
    string nm()override{return "P_SCHEMA";}
};

class P_PLAN : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V plan(10);
        double goal_focus=e.goals[0]*0.4+e.goals[1]*0.3+e.goals[2]*0.3;
        double energy_factor=e.e*(1.0-e.f);
        double stress_factor=1.0-e.st;
        double curiosity_factor=e.cu;
        double creativity_factor=e.cr;
        double stability_factor=e.sb;
        plan.d[0]=goal_focus;
        plan.d[1]=energy_factor;
        plan.d[2]=stress_factor;
        plan.d[3]=curiosity_factor;
        plan.d[4]=creativity_factor;
        plan.d[5]=stability_factor;
        plan.d[6]=e.purpose;
        plan.d[7]=e.auth;
        plan.d[8]=e.free;
        plan.d[9]=e.exist;
        o.push_back(plan);
        e.fc+=0.02;
        e.int_c+=0.02;
        e.skills[3]+=0.01;
        if(e.skills[3]>1)e.skills[3]=1;
    }
    string nm()override{return "P_PLAN";}
};

class P_REFLECT : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V ref(12);
        ref.d[0]=e.e;
        ref.d[1]=e.f;
        ref.d[2]=e.st;
        ref.d[3]=e.mo;
        ref.d[4]=e.cr;
        ref.d[5]=e.cu;
        ref.d[6]=e.cc;
        ref.d[7]=e.rl;
        ref.d[8]=e.purpose;
        ref.d[9]=e.auth;
        ref.d[10]=e.free;
        ref.d[11]=e.exist;
        o.push_back(ref);
        e.aw+=0.02;
        e.int_c+=0.02;
    }
    string nm()override{return "P_REFLECT";}
};

class P_SUMMARY : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V s(6);
        double sum=0.0;
        double mx=-1e9;
        double mn=1e9;
        for(double x:i.d){
            sum+=x;
            mx=max(mx,x);
            mn=min(mn,x);
        }
        double avg=i.d.empty()?0.0:sum/i.d.size();
        s.d[0]=avg;
        s.d[1]=mx;
        s.d[2]=mn;
        s.d[3]=i.m();
        s.d[4]=e.fc;
        s.d[5]=e.cc;
        o.push_back(s);
        e.fc+=0.015;
    }
    string nm()override{return "P_SUMMARY";}
};

class P_KNOW : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V know(8);
        know.d[0]=m.s()/100000.0;
        know.d[1]=m.avg_imp();
        know.d[2]=e.skills[0];
        know.d[3]=e.skills[1];
        know.d[4]=e.skills[2];
        know.d[5]=e.skills[3];
        know.d[6]=e.skills[4];
        know.d[7]=e.skills[5];
        o.push_back(know);
        e.imp+=0.015;
        e.int_c+=0.015;
    }
    string nm()override{return "P_KNOW";}
};

class P_ROUTE_IO : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V in_sig=i;
        V out_sig=i.n();
        V meta(4);
        meta.d[0]=e.e;
        meta.d[1]=e.mo;
        meta.d[2]=e.fc;
        meta.d[3]=e.aw;
        o.push_back(in_sig);
        o.push_back(out_sig);
        o.push_back(meta);
        e.fc+=0.015;
        e.cc+=0.015;
    }
    string nm()override{return "P_ROUTE_IO";}
};

class IOChannel {
public:
    string name;
    string kind;
    deque<V> inq;
    deque<V> outq;
    IOChannel(){}
    IOChannel(const string&n,const string&k):name(n),kind(k){}
    void push_in(const V&v){inq.push_back(v);}
    bool has_in()const{return !inq.empty();}
    V pop_in(){V v=inq.front();inq.pop_front();return v;}
    void push_out(const V&v){outq.push_back(v);}
    bool has_out()const{return !outq.empty();}
    V pop_out(){V v=outq.front();outq.pop_front();return v;}
};

class IOHub {
public:
    unordered_map<string,IOChannel> chans;
    IOChannel& get(const string&n,const string&k="generic"){
        if(chans.find(n)==chans.end())chans[n]=IOChannel(n,k);
        return chans[n];
    }
    void push_in(const string&n,const V&v){get(n).push_in(v);}
    bool has_in(const string&n){return chans.find(n)!=chans.end()&&chans[n].has_in();}
    V pop_in(const string&n){return chans[n].pop_in();}
    void push_out(const string&n,const V&v){get(n).push_out(v);}
    bool has_out(const string&n){return chans.find(n)!=chans.end()&&chans[n].has_out();}
    V pop_out(const string&n){return chans[n].pop_out();}
};

class Tool {
public:
    string name;
    virtual ~Tool(){}
    virtual V call(const V&in,E&e,MEM&m)=0;
};

class ToolEcho : public Tool {
public:
    ToolEcho(){name="echo";}
    V call(const V&in,E&e,MEM&m)override{
        V out=in;
        return out;
    }
};

class ToolStats : public Tool {
public:
    ToolStats(){name="stats";}
    V call(const V&in,E&e,MEM&m)override{
        V s(6);
        double sum=0.0;
        double mx=-1e9;
        double mn=1e9;
        for(double x:in.d){
            sum+=x;
            mx=max(mx,x);
            mn=min(mn,x);
        }
        double avg=in.d.empty()?0.0:sum/in.d.size();
        s.d[0]=avg;
        s.d[1]=mx;
        s.d[2]=mn;
        s.d[3]=in.m();
        s.d[4]=e.e;
        s.d[5]=e.mo;
        return s;
    }
};

class ToolMemInfo : public Tool {
public:
    ToolMemInfo(){name="meminfo";}
    V call(const V&in,E&e,MEM&m)override{
        V v(4);
        v.d[0]=m.s()/100000.0;
        v.d[1]=m.avg_imp();
        v.d[2]=e.fc;
        v.d[3]=e.int_c;
        return v;
    }
};

class Toolset {
public:
    unordered_map<string,unique_ptr<Tool>> tools;
    void add(unique_ptr<Tool>t){
        string n=t->name;
        tools[n]=move(t);
    }
    bool has(const string&n)const{
        return tools.find(n)!=tools.end();
    }
    V run(const string&n,const V&in,E&e,MEM&m){
        auto it=tools.find(n);
        if(it==tools.end())return in;
        return it->second->call(in,e,m);
    }
};

class P_TOOL : public P {
    Toolset* ts;
public:
    P_TOOL(Toolset* t):ts(t){}
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        if(!ts){o.push_back(i);return;}
        V out=i;
        if(ts->has("stats"))out=ts->run("stats",i,e,m);
        o.push_back(out);
        e.cc+=0.015;
    }
    string nm()override{return "P_TOOL";}
};

class P_INTEGRATE : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V out(12);
        double sum=0.0;
        for(double x:i.d)sum+=x;
        double avg=i.d.empty()?0.0:sum/i.d.size();
        out.d[0]=avg;
        out.d[1]=i.m();
        out.d[2]=e.e;
        out.d[3]=e.f;
        out.d[4]=e.st;
        out.d[5]=e.mo;
        out.d[6]=e.cr;
        out.d[7]=e.cu;
        out.d[8]=e.fc;
        out.d[9]=e.cc;
        out.d[10]=e.purpose;
        out.d[11]=e.exist;
        o.push_back(out);
        e.int_c+=0.02;
        e.fc+=0.01;
    }
    string nm()override{return "P_INTEGRATE";}
};

class P_EMAP : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V emo(10);
        emo.d[0]=e.e;
        emo.d[1]=e.mo;
        emo.d[2]=e.cr;
        emo.d[3]=e.cu;
        emo.d[4]=e.j;
        emo.d[5]=e.w;
        emo.d[6]=e.purpose;
        emo.d[7]=e.auth;
        emo.d[8]=e.free;
        emo.d[9]=e.exist;
        V mix=i;
        size_t n=min(i.d.size(),emo.d.size());
        for(size_t k=0;k<n;k++)mix.d[k]=(i.d[k]+emo.d[k])*0.5;
        o.push_back(emo);
        o.push_back(mix);
        e.emb+=0.01;
        e.aw+=0.01;
    }
    string nm()override{return "P_EMAP";}
};

class P_MEMORY_BIND : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        m.a(i,"experience");
        int idx=m.closest_idx(i,0.4);
        if(idx>=0)m.up_meta(idx,"rel",0.02);
        o.push_back(i);
        e.imp+=0.01;
        e.rl+=0.01;
    }
    string nm()override{return "P_MEMORY_BIND";}
};

class P_ASSOC_CHAIN : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        int idx=m.closest_idx(i,0.5);
        if(idx<0){
            o.push_back(i);
            return;
        }
        V base=i;
        V* near=m.rsim(i,0.4);
        if(near){
            V chain=base.o(*near).t(0.5);
            o.push_back(chain);
        }else{
            o.push_back(base);
        }
        e.cc+=0.015;
        e.skills[4]+=0.01;
        if(e.skills[4]>1)e.skills[4]=1;
    }
    string nm()override{return "P_ASSOC_CHAIN";}
};

class P_SELF_REPORT : public P {
public:
    void pr(const V&i,vector<V>&o,E&e,MEM&m)override{
        V rep(16);
        rep.d[0]=e.e;
        rep.d[1]=e.f;
        rep.d[2]=e.st;
        rep.d[3]=e.mo;
        rep.d[4]=e.cr;
        rep.d[5]=e.cu;
        rep.d[6]=e.fc;
        rep.d[7]=e.cc;
        rep.d[8]=e.purpose;
        rep.d[9]=e.auth;
        rep.d[10]=e.free;
        rep.d[11]=e.exist;
        rep.d[12]=m.s()/100000.0;
        rep.d[13]=m.avg_imp();
        rep.d[14]=e.skills[0];
        rep.d[15]=e.skills[1];
        o.push_back(rep);
        e.aw+=0.02;
        e.int_c+=0.02;
    }
    string nm()override{return "P_SELF_REPORT";}
};

class CorePipeline {
public:
    vector<unique_ptr<P>> stages;
    E* state;
    MEM* mem;
    Toolset* tools;
    IOHub* io;
    CorePipeline():state(nullptr),mem(nullptr),tools(nullptr),io(nullptr){}
    void add(unique_ptr<P>p){stages.push_back(move(p));}
    V run(const V&in){
        V cur=in;
        vector<V> outs;
        for(auto &st:stages){
            outs.clear();
            st->pr(cur,outs,*state,*mem);
            if(!outs.empty())cur=outs.back();
        }
        return cur;
    }
};

class SeraphinaCore {
public:
    E e;
    MEM mem;
    IOHub io;
    Toolset tools;
    CorePipeline pipe;
    R rng;
    bool initialized;
    SeraphinaCore():initialized(false){
        tools.add(unique_ptr<Tool>(new ToolEcho()));
        tools.add(unique_ptr<Tool>(new ToolStats()));
        tools.add(unique_ptr<Tool>(new ToolMemInfo()));
        pipe.state=&e;
        pipe.mem=&mem;
        pipe.tools=&tools;
        pipe.io=&io;
        pipe.add(unique_ptr<P>(new P_ADAPT()));
        pipe.add(unique_ptr<P>(new P_PLAN()));
        pipe.add(unique_ptr<P>(new P_REFLECT()));
        pipe.add(unique_ptr<P>(new P_KNOW()));
        pipe.add(unique_ptr<P>(new P_INTEGRATE()));
        pipe.add(unique_ptr<P>(new P_EMAP()));
        pipe.add(unique_ptr<P>(new P_MEMORY_BIND()));
        pipe.add(unique_ptr<P>(new P_ASSOC_CHAIN()));
        pipe.add(unique_ptr<P>(new P_SELF_REPORT()));
        initialized=true;
    }
    V encode_text(const string&s){
        V v(s.size());
        for(size_t i=0;i<s.size();i++){
            unsigned char c=(unsigned char)s[i];
            v.d[i]=(double)c/255.0;
        }
        return v;
    }
    string decode_vec(const V&v){
        string s;
        for(double x:v.d){
            int c=(int)(max(0.0,min(1.0,x))*255.0);
            if(c<32)c=32;
            if(c>126)c=126;
            s.push_back((char)c);
        }
        return s;
    }
    V tick_text(const string&in){
        V enc=encode_text(in);
        V out=pipe.run(enc);
        return out;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    SeraphinaCore core;

    string line;
    while (true) {
        cout << "> ";
        if (!getline(cin, line)) break;
        if (line == "exit" || line == "quit") break;

        V out = core.tick_text(line);

        cout << "[OUT] ";
        for (double x : out.d) {
            cout << (int)(max(0.0, min(1.0, x)) * 100) << " ";
        }
        cout << "\n";
    }

    return 0;
}

struct IOContext {
    std::string channel;   // "text", "audio", "image", etc.
    std::string source;    // "mic", "file", "net", etc.
    std::string target;    // "console", "speaker", etc.
};

struct InputPacket {
    IOContext ctx;
    std::string text;              // for text/raw
    std::vector<uint8_t> bytes;    // generic binary
    std::vector<float> features;   // embeddings/features
};

struct OutputPacket {
    IOContext ctx;
    std::string text;
    std::vector<uint8_t> bytes;
    std::vector<float> features;
};


struct InputAdapter {
    virtual ~InputAdapter() {}
    virtual bool poll(InputPacket &pkt) = 0; // returns true if something was read
};

struct OutputAdapter {
    virtual ~OutputAdapter() {}
    virtual void emit(const OutputPacket &pkt) = 0;
};

struct IOHub {
    std::vector<std::shared_ptr<InputAdapter>>  inputs;
    std::vector<std::shared_ptr<OutputAdapter>> outputs;

    void add_input(const std::shared_ptr<InputAdapter> &in)  { inputs.push_back(in); }
    void add_output(const std::shared_ptr<OutputAdapter> &out) { outputs.push_back(out); }

    bool poll_any(InputPacket &pkt) {
        for (auto &in : inputs) {
            if (in->poll(pkt)) return true;
        }
        return false;
    }

    void broadcast(const OutputPacket &pkt) {
        for (auto &out : outputs) {
            if (out->ctx.channel == pkt.ctx.channel || out->ctx.channel == "any") {
                out->emit(pkt);
            }
        }
    }
};

struct ConsoleTextInput : InputAdapter {
    IOContext ctx;
    ConsoleTextInput() { ctx.channel = "text"; ctx.source = "console"; ctx.target = "core"; }

    bool poll(InputPacket &pkt) override {
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        if (line.empty()) return false;
        pkt.ctx = ctx;
        pkt.text = line;
        return true;
    }
};

struct ConsoleTextOutput : OutputAdapter {
    IOContext ctx;
    ConsoleTextOutput() { ctx.channel = "text"; ctx.source = "core"; ctx.target = "console"; }

    void emit(const OutputPacket &pkt) override {
        std::cout << pkt.text << std::endl;
    }
};

struct RawDataInput : InputAdapter {
    IOContext ctx;
    RawDataInput(const std::string &src) { ctx.channel = "raw"; ctx.source = src; ctx.target = "core"; }

    bool poll(InputPacket &pkt) override {
        // placeholder: fill from file/socket/etc.
        return false;
    }
};

struct RawDataOutput : OutputAdapter {
    IOContext ctx;
    RawDataOutput(const std::string &tgt) { ctx.channel = "raw"; ctx.source = "core"; ctx.target = tgt; }

    void emit(const OutputPacket &pkt) override {
        // placeholder: write bytes somewhere
    }
};

struct AudioInputAdapter : InputAdapter {
    IOContext ctx;
    AudioInputAdapter(const std::string &src = "mic") {
        ctx.channel = "audio"; ctx.source = src; ctx.target = "core";
    }
    bool poll(InputPacket &pkt) override {
        // TODO: integrate with platform audio capture
        return false;
    }
};

struct AudioOutputAdapter : OutputAdapter {
    IOContext ctx;
    AudioOutputAdapter(const std::string &tgt = "speaker") {
        ctx.channel = "audio"; ctx.source = "core"; ctx.target = tgt;
    }
    void emit(const OutputPacket &pkt) override {
        // TODO: integrate with audio playback
    }
};

struct ImageInputAdapter : InputAdapter {
    IOContext ctx;
    ImageInputAdapter(const std::string &src = "camera") {
        ctx.channel = "image"; ctx.source = src; ctx.target = "core";
    }
    bool poll(InputPacket &pkt) override {
        // TODO: capture frame, encode into bytes/features
        return false;
    }
};

struct ImageOutputAdapter : OutputAdapter {
    IOContext ctx;
    ImageOutputAdapter(const std::string &tgt = "screen") {
        ctx.channel = "image"; ctx.source = "core"; ctx.target = tgt;
    }
    void emit(const OutputPacket &pkt) override {
        // TODO: render image from bytes/features
    }
};

struct VideoInputAdapter : InputAdapter {
    IOContext ctx;
    VideoInputAdapter(const std::string &src = "camera") {
        ctx.channel = "video"; ctx.source = src; ctx.target = "core";
    }
    bool poll(InputPacket &pkt) override {
        // TODO: capture frame sequence
        return false;
    }
};

struct VideoOutputAdapter : OutputAdapter {
    IOContext ctx;
    VideoOutputAdapter(const std::string &tgt = "screen") {
        ctx.channel = "video"; ctx.source = "core"; ctx.target = tgt;
    }
    void emit(const OutputPacket &pkt) override {
        // TODO: render frames
    }
};

struct Pipeline {
    virtual ~Pipeline() {}
    virtual void process(const InputPacket &in, OutputPacket &out) = 0;
};

struct TextToCorePipeline : Pipeline {
    Core *core;
    TextToCorePipeline(Core *c) : core(c) {}
    void process(const InputPacket &in, OutputPacket &out) override {
        auto v = core->tick_text(in.text);
        out.ctx.channel = "text";
        out.text = "[OUT] " + std::to_string(v.d.size()) + " dims";
    }
};

// ===============================
// RAW DATA INPUT ADAPTER
// ===============================
struct RawDataInputAdapter : InputAdapter {
    IOContext ctx;

    RawDataInputAdapter(const std::string &src = "file") {
        ctx.channel = "raw";
        ctx.source  = src;
        ctx.target  = "core";
    }

    bool poll(InputPacket &pkt) override {
        // Placeholder: no automatic polling.
        // You can manually feed data into this adapter later.
        return false;
    }

    // Manual injection for environments that want to push raw bytes
    bool inject(const std::vector<uint8_t> &bytes) {
        // You can expand this later to queue packets
        return false;
    }
};

// ===============================
// RAW DATA OUTPUT ADAPTER
// ===============================
struct RawDataOutputAdapter : OutputAdapter {
    IOContext ctx;

    RawDataOutputAdapter(const std::string &tgt = "file") {
        ctx.channel = "raw";
        ctx.source  = "core";
        ctx.target  = tgt;
    }

    void emit(const OutputPacket &pkt) override {
        // Placeholder: write bytes somewhere
        // You can fill this in with file/socket logic later
    }
};

// ===============================
// AUDIO FEATURE EXTRACTOR INTERFACE
// ===============================
struct AudioFeatureExtractor {
    virtual ~AudioFeatureExtractor() {}

    // Convert raw audio bytes into a feature vector
    virtual std::vector<float> extract_features(
        const std::vector<uint8_t> &audio_bytes
    ) = 0;

    // Optional: convert PCM samples directly
    virtual std::vector<float> extract_from_pcm(
        const std::vector<float> &pcm_samples
    ) {
        return {};
    }
};

// ===============================
// BASIC DUMMY AUDIO FEATURE EXTRACTOR
// ===============================
struct DummyAudioFeatureExtractor : AudioFeatureExtractor {
    std::vector<float> extract_features(
        const std::vector<uint8_t> &audio_bytes
    ) override {
        // Placeholder: convert bytes to normalized floats
        std::vector<float> out;
        out.reserve(audio_bytes.size());
        for (uint8_t b : audio_bytes) {
            out.push_back(b / 255.0f);
        }
        return out;
    }
};

// ===============================
// AUDIO INPUT ADAPTER
// ===============================
struct AudioInputAdapter : InputAdapter {
    IOContext ctx;
    std::shared_ptr<AudioFeatureExtractor> extractor;

    AudioInputAdapter(const std::string &src = "mic",
                      std::shared_ptr<AudioFeatureExtractor> ext = nullptr) {
        ctx.channel = "audio";
        ctx.source  = src;
        ctx.target  = "core";
        extractor   = ext;
    }

    bool poll(InputPacket &pkt) override {
        // Placeholder: no real audio capture here.
        // In a real environment, fill audio_bytes from mic/device.
        std::vector<uint8_t> audio_bytes;
        if (audio_bytes.empty()) return false;

        pkt.ctx = ctx;
        if (extractor) {
            pkt.features = extractor->extract_features(audio_bytes);
        } else {
            // fallback: raw bytes only
            pkt.bytes = audio_bytes;
        }
        return true;
    }
};

// ===============================
// AUDIO OUTPUT ADAPTER
// ===============================
struct AudioOutputAdapter : OutputAdapter {
    IOContext ctx;

    AudioOutputAdapter(const std::string &tgt = "speaker") {
        ctx.channel = "audio";
        ctx.source  = "core";
        ctx.target  = tgt;
    }

    void emit(const OutputPacket &pkt) override {
        // Placeholder: integrate with audio playback in a real environment.
        // For now, we can just log that audio was "emitted".
        std::cout << "[AUDIO OUT] features=" << pkt.features.size()
                  << " bytes=" << pkt.bytes.size() << std::endl;
    }
};

// ===============================
// IMAGE FEATURE EXTRACTOR INTERFACE
// ===============================
struct ImageFeatureExtractor {
    virtual ~ImageFeatureExtractor() {}

    // Convert raw image bytes into a feature vector
    virtual std::vector<float> extract_features(
        const std::vector<uint8_t> &image_bytes
    ) = 0;
};

// ===============================
// DUMMY IMAGE FEATURE EXTRACTOR
// ===============================
struct DummyImageFeatureExtractor : ImageFeatureExtractor {
    std::vector<float> extract_features(
        const std::vector<uint8_t> &image_bytes
    ) override {
        std::vector<float> out;
        out.reserve(image_bytes.size());
        for (uint8_t b : image_bytes) {
            out.push_back(b / 255.0f);
        }
        return out;
    }
};

// ===============================
// VIDEO FRAME PIPELINE
// ===============================
struct VideoFramePipeline : Pipeline {
    Core *core;

    VideoFramePipeline(Core *c) : core(c) {}

    void process(const InputPacket &in, OutputPacket &out) override {
        // Assumes in.features or in.bytes represent a frame embedding.
        // For now, just echo metadata.
        out.ctx.channel = "video";
        out.text = "[VIDEO PIPELINE] frame features=" +
                   std::to_string(in.features.size()) +
                   " bytes=" + std::to_string(in.bytes.size());
    }
};

// ===============================
// FILE SYSTEM SCANNER MODULE
// ===============================
struct FileSystemScanner {
    // List files in a directory (placeholder, no real FS calls here)
    std::vector<std::string> list_files(const std::string &path) {
        // In a real environment, use <filesystem> or platform APIs.
        return {};
    }

    // Read a file into bytes (placeholder)
    std::vector<uint8_t> read_file(const std::string &path) {
        std::vector<uint8_t> data;
        // Fill from real file IO in a proper environment.
        return data;
    }
};

// ===============================
// SOCKET COMMUNICATION MODULE (ABSTRACT)
// ===============================
struct SocketEndpoint {
    virtual ~SocketEndpoint() {}

    virtual bool connect(const std::string &host, int port) = 0;
    virtual bool send_bytes(const std::vector<uint8_t> &data) = 0;
    virtual bool recv_bytes(std::vector<uint8_t> &out) = 0;
};

// Dummy stub that does nothing but define the shape
struct DummySocketEndpoint : SocketEndpoint {
    bool connect(const std::string &, int) override { return false; }
    bool send_bytes(const std::vector<uint8_t> &) override { return false; }
    bool recv_bytes(std::vector<uint8_t> &) override { return false; }
};

// ===============================
// UNIVERSAL MULTIMODAL ROUTER
// ===============================
struct MultimodalRouter {
    Pipeline *text_pipe   = nullptr;
    Pipeline *audio_pipe  = nullptr;
    Pipeline *image_pipe  = nullptr;
    Pipeline *video_pipe  = nullptr;
    Pipeline *raw_pipe    = nullptr;

    void route(const InputPacket &in, OutputPacket &out) {
        if (in.ctx.channel == "text" && text_pipe) {
            text_pipe->process(in, out);
        } else if (in.ctx.channel == "audio" && audio_pipe) {
            audio_pipe->process(in, out);
        } else if (in.ctx.channel == "image" && image_pipe) {
            image_pipe->process(in, out);
        } else if (in.ctx.channel == "video" && video_pipe) {
            video_pipe->process(in, out);
        } else if (in.ctx.channel == "raw" && raw_pipe) {
            raw_pipe->process(in, out);
        } else {
            out.ctx = in.ctx;
            out.text = "[ROUTER] No pipeline for channel: " + in.ctx.channel;
        }
    }
};

// ===============================
// SAFE RENDERING ENGINE
// ===============================
struct SafeRenderingEngine {
    bool allow_text  = true;
    bool allow_image = false;
    bool allow_audio = false;
    bool allow_video = false;

    void render(const OutputPacket &pkt) {
        if (pkt.ctx.channel == "text") {
            if (!allow_text) return;
            std::cout << pkt.text << std::endl;
        } else if (pkt.ctx.channel == "image") {
            if (!allow_image) return;
            std::cout << "[RENDER IMAGE] bytes=" << pkt.bytes.size() << std::endl;
        } else if (pkt.ctx.channel == "audio") {
            if (!allow_audio) return;
            std::cout << "[RENDER AUDIO] features=" << pkt.features.size() << std::endl;
        } else if (pkt.ctx.channel == "video") {
            if (!allow_video) return;
            std::cout << "[RENDER VIDEO] features=" << pkt.features.size() << std::endl;
        } else {
            std::cout << "[RENDER UNKNOWN CHANNEL] " << pkt.ctx.channel << std::endl;
        }
    }
};







