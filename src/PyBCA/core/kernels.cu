// CUDA integer kernels for Volta (sm_70) and newer. No PyTorch C++ ABI needed.
#include <cuda_runtime.h>
#include <stdint.h>

__device__ uint32_t philox(const uint32_t* key, uint32_t index, uint64_t step,
                           uint32_t rule, uint32_t domain) {
    uint32_t a=index, b=uint32_t(step), c=uint32_t(step>>32), d=(rule<<8)|domain;
    uint32_t k0=key[0], k1=key[1];
    for(int i=0;i<10;i++) {
        uint64_t p=uint64_t(a)*0xD2511F53u, q=uint64_t(c)*0xCD9E8D57u;
        a=uint32_t(q>>32)^b^k0; b=uint32_t(q);
        c=uint32_t(p>>32)^d^k1; d=uint32_t(p);
        k0+=0x9E3779B9u; k1+=0xBB67AE85u;
    }
    return a;
}
__device__ float uniform(const uint32_t* key, uint32_t index, uint64_t step,
                         uint32_t rule, uint32_t domain) {
    return float(philox(key,index,step,rule,domain)>>8)*0x1p-24f;
}
__global__ void order_kernel(int* order,const uint32_t* keys,int t,int n,uint64_t step) {
    int trial=blockIdx.x*blockDim.x+threadIdx.x;
    if(trial>=t) return;
    int* row=order+trial*n;
    for(int r=0;r<n;r++) row[r]=r;
    for(int i=n-1;i>0;i--) {
        uint32_t bound=i+1, threshold=(-bound)%bound, attempt=0;
        uint32_t u=philox(keys+trial*2,i,step,attempt,3);
        while(u<threshold) u=philox(keys+trial*2,i,step,++attempt,3);
        int j=u%bound, tmp=row[i]; row[i]=row[j]; row[j]=tmp;
    }
}
__global__ void match_kernel(const int8_t* cells,const int8_t* rules,const float* probs,
    const uint32_t* keys,bool* mask,int* candidates,int* counts,int t,int n,int h,int w,
    int cap,uint64_t step,float global,int independent,int prob_stride) {
    int hw=h*w;
    int64_t size=int64_t(t)*n*hw;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size;i+=int64_t(blockDim.x)*gridDim.x) {
        int cell=i%hw, r=(i/hw)%n, trial=i/(int64_t(hw)*n), y=cell/w,x=cell%w;
        const int8_t* pre=rules+r*18;
        bool ok=true;
        // Start with the center for sparse BCA circuits.
        if(cells[trial*hw+cell]!=pre[4]) ok=false;
        for(int j=0;ok && j<9;j++) {
            if(j==4 || ((j==0||j==2||j==6||j==8) && pre[j]==0)) continue;
            int yy=y+j/3-1, xx=x+j%3-1;
            int v=(yy>=0&&yy<h&&xx>=0&&xx<w)?cells[trial*hw+yy*w+xx]:0;
            ok=(v==pre[j]);
        }
        if(ok && independent) {
            ok=uniform(keys+trial*2,cell,step,r,1)<global
                && uniform(keys+trial*2,cell,step,r,2)<probs[trial*prob_stride+r];
        }
        mask[i]=ok;
        if(ok && independent) {
            int pos=atomicAdd(counts+trial*n+r,1);
            if(pos<cap) candidates[(trial*n+r)*cap+pos]=cell;
        }
    }
}
__global__ void pack_kernel(const bool* mask,int* candidates,int* counts,int t,int n,int hw,int cap) {
    int64_t size=int64_t(t)*n*hw;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size;i+=int64_t(blockDim.x)*gridDim.x) {
        if(mask[i]) {
            int row=i/hw, cell=i%hw;
            int pos=atomicAdd(counts+row,1);
            if(pos<cap) candidates[row*cap+pos]=cell;
        }
    }
}
__global__ void conflict_kernel(const bool* mask,const bool* written,const int8_t* rules,
    const int* candidates,const int* counts,const int* order,bool* accepted,int* rule_counts,
    int n,int h,int w,int cap,int position,int record) {
    int trial=blockIdx.y, hw=h*w, r=order[trial*n+position], row=trial*n+r;
    int count=counts[row], length=count>cap?hw:count;
    const int8_t* pre=rules+r*18; const int8_t* post=pre+9;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<length;i+=blockDim.x*gridDim.x) {
        int center=count>cap?i:candidates[row*cap+i];
        if(!mask[int64_t(row)*hw+center]) continue;
        int y=center/w,x=center%w; bool keep=true;
        for(int q=0;keep && q<9;q++) {
            if(pre[q]==post[q]||post[q]==-1) continue;
            int yy=y+q/3-1,xx=x+q%3-1;
            if(yy<0||yy>=h||xx<0||xx>=w) continue;
            if(written[trial*hw+yy*w+xx]) {keep=false;break;}
            for(int j=0;j<9;j++) {
                if(j==q||pre[j]==post[j]||post[j]==-1) continue;
                int cy=yy-(j/3-1),cx=xx-(j%3-1);
                if(cy>=0&&cy<h&&cx>=0&&cx<w&&mask[int64_t(row)*hw+cy*w+cx]) {keep=false;break;}
            }
        }
        accepted[trial*hw+center]=keep;
        if(keep && record) atomicAdd(rule_counts+row,1);
    }
}
__global__ void write_kernel(bool* mask,bool* written,int8_t* cells,const int8_t* rules,
    const int* candidates,const int* counts,const int* order,const bool* accepted,
    int n,int h,int w,int cap,int position) {
    int trial=blockIdx.y,hw=h*w,r=order[trial*n+position],row=trial*n+r;
    int count=counts[row],length=count>cap?hw:count;
    const int8_t* pre=rules+r*18;const int8_t* post=pre+9;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<length;i+=blockDim.x*gridDim.x) {
        int center=count>cap?i:candidates[row*cap+i];
        int64_t loc=int64_t(row)*hw+center;
        if(!mask[loc]) continue;
        bool keep=accepted[trial*hw+center]; mask[loc]=keep;
        if(!keep) continue;
        int y=center/w,x=center%w;
        // Surviving targets are unique, so no atomic write or winner selection.
        for(int q=0;q<9;q++) {
            if(pre[q]==post[q]||post[q]==-1) continue;
            int yy=y+q/3-1,xx=x+q%3-1;
            if(yy>=0&&yy<h&&xx>=0&&xx<w) {
                int dst=trial*hw+yy*w+xx; cells[dst]=post[q];written[dst]=true;
            }
        }
    }
}
__global__ void event_match(const int8_t* cells,const int64_t* ev,const float* probs,
    const uint32_t* keys,bool* hits,int t,int e,int hw,uint64_t step) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=t*e) return;
    int trial=i/e,event=i%e; const int64_t* row=ev+event*7;
    hits[i]=row[0]>=0&&(row[5]<0||step>=uint64_t(row[5]))&&(row[6]<0||step<=uint64_t(row[6]))
        &&cells[trial*hw+row[0]]==row[1]&&uniform(keys+trial*2,event,step,0,4)<probs[event];
}
__global__ void event_write(int8_t* cells,bool* written,const int64_t* ev,const bool* hits,int t,int e,int hw) {
    int trial=blockIdx.x*blockDim.x+threadIdx.x;if(trial>=t)return;
    for(int j=0;j<e;j++) if(hits[trial*e+j]) {
        int dst=trial*hw+ev[j*7+2];cells[dst]=ev[j*7+3];written[dst]=true;
    }
}
__global__ void state_kernel(int8_t* cells,bool* written,const int8_t* lut,int size) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x) {
        int8_t old=cells[i],val=lut[int(old)+128];cells[i]=val;written[i]|=(val!=old);
    }
}

// One stable C entry point. Pointers and dimensions are marshalled by the
// validated Python wrapper; all launches use PyTorch's current CUDA stream.
extern "C" int pybca_launch(int op,const int64_t* a,float global,void* raw_stream) {
    cudaStream_t stream=(cudaStream_t)raw_stream;
    auto cells=(int8_t*)a[0];auto rules=(int8_t*)a[1];auto probs=(float*)a[2];auto keys=(uint32_t*)a[3];
    auto mask=(bool*)a[4];auto candidates=(int*)a[5];auto counts=(int*)a[6];
    auto written=(bool*)a[7];auto accepted=(bool*)a[8];auto order=(int*)a[9];auto rc=(int*)a[10];
    int t=a[11],n=a[12],h=a[13],w=a[14],cap=a[15];uint64_t step=a[16];
    int independent=a[17],stride=a[18],record=a[19];
    if(op==0) {
        match_kernel<<<512,256,0,stream>>>(cells,rules,probs,keys,mask,candidates,counts,t,n,h,w,cap,step,global,independent,stride);
    } else if(op==1) {
        pack_kernel<<<512,256,0,stream>>>(mask,candidates,counts,t,n,h*w,cap);
    } else if(op==2) {
        if(independent) order_kernel<<<(t+127)/128,128,0,stream>>>(order,keys,t,n,step);
        for(int p=0;p<n;p++) {
            conflict_kernel<<<dim3(32,t),128,0,stream>>>(mask,written,rules,candidates,counts,order,accepted,rc,n,h,w,cap,p,record);
            write_kernel<<<dim3(32,t),128,0,stream>>>(mask,written,cells,rules,candidates,counts,order,accepted,n,h,w,cap,p);
        }
    } else if(op==3) {
        auto ev=(int64_t*)a[20];auto ep=(float*)a[21];auto hits=(bool*)a[22];int e=a[23];
        if(e) {
            event_match<<<(t*e+127)/128,128,0,stream>>>(cells,ev,ep,keys,hits,t,e,h*w,step);
            event_write<<<(t+127)/128,128,0,stream>>>(cells,written,ev,hits,t,e,h*w);
        }
    } else if(op==4) {
        state_kernel<<<512,256,0,stream>>>(cells,written,(int8_t*)a[24],t*h*w);
    }
    return int(cudaGetLastError());
}
