#include<iostream>
#include<cmath>
#include<random>

std::mt19937 mt_rand(0);
std::uniform_real_distribution<double> d(0.0,1.0);

double rand_u(){
  return d(mt_rand);
}

   
class JSQd {
  double *X; // X[i] = number of queues with i or more jobs 
  int N; // total number of queues
  double rho; // system's load
  double oneOverN;
  int K; // queues capacities
  int d; // number of queues to sample from
  int total_number_jobs;
  bool withoutReplacement;  // to pick jobs without replacement 
  bool only_print_total_number_of_jobs; 
  
 public:
  JSQd(int N, double rho, int K, int d, bool withoutReplacement,
       bool only_print_total_number_of_jobs)
    : N(N), rho(rho), K(K), d(d), withoutReplacement(withoutReplacement),
      only_print_total_number_of_jobs(only_print_total_number_of_jobs) {
    oneOverN = 1./(double)N;
    X = new double[K+1];
    for(int i=0;i<K+1;i++) X[i] = 0;
    X[0] = 1;
    if (withoutReplacement){
      std::cerr << "without replacement not implemented for now\n";
    }
    total_number_jobs = N;
  }
  void print() {
    if(only_print_total_number_of_jobs){
      std::cout << total_number_jobs/(double)N << "\n";
    }
    else
      {
	for(int i=0;i<K;i++) std::cout << X[i] << " ";
	std::cout << "\n";
      }
  }
  void simulate(int T){
    simulate_twoChoice(T);
  }
  void simulate_twoChoice(int T){
    if (withoutReplacement)
      simulate_twoChoice_withoutReplacement(T);
    else
      simulate_twoChoice_withReplacement(T);
  }
  void simulate_twoChoice_withReplacement(int T) {
    for(int t=0;t<T;t++){
      double u = rand_u();
      //std::cerr << u << "\n";
      if (u < 1/(1+rho) ) { // departure
	//std::cout << "departure\n";
	double u = rand_u();
	int i=0;
	//while( i<K && X[i]-X[i+1] < u) {u += X[i+1]-X[i]; i++;}
	while( i<K && u < X[i+1] ) {i++; }
	if (i>0) {X[i]-=oneOverN; total_number_jobs--;}
      }
      else{                 // arrival
	//std::cout << "arrival @ ";
	double u = rand_u();
	
	int i=0;
	while( i<K && u < pow(X[i+1],d)) { i++;}
	if (i<K) {X[i+1]+=oneOverN; total_number_jobs++;}
      }
    }
  }
  void simulate_oneChoice(int T) {
    for(int t=0;t<T;t++){
      double u = rand_u();
      if (u < 1/(1+rho) ) { // departure
	double u = rand_u();
	int i=0;
	while( i<K && u < X[i+1] ) {i++; }
	if (i>0) {X[i]-=oneOverN; total_number_jobs--;}
      }
      else{                 // arrival
	double u = rand_u();
	double v = rand_u(); (void) v;
	int i=0;
	//while( i<K && X[i]-X[i+1] < u) {u += X[i+1]-X[i]; i++;}
	while( i<K && u < X[i+1] ) { i++; }
	if (i<K) {X[i+1]+=oneOverN; total_number_jobs++;}
      }
    }
  }
  void simulate_twoChoice_withoutReplacement(int T) {
    for(int t=0;t<T;t++){
      double u = rand_u();
      if (u < 1/(1+rho) ) { // departure
	//std::cout << "departure\n";
	double u = rand_u();
	int i=0;
	while( i<K && X[i]-X[i+1] < u) {u += X[i+1]-X[i]; i++;}
	if (i>0) {X[i]-=oneOverN; total_number_jobs--;}
      }
      else{                 // arrival
	//std::cout << "arrival @ ";
	double u = rand_u();
	int i=0;
	while( i<K && (X[i]-X[i+1])*(X[i+1]+X[i]+oneOverN)/(1-oneOverN) < u) {
	  u += -(X[i]-X[i+1])*(X[i+1]+X[i]+oneOverN)/(1-oneOverN); i++;}
	if (i<K) {X[i+1]+=oneOverN; total_number_jobs++;}
	//std::cout << i<< " (" << u<<")\n";
      }
    }
  }
  void steady_state(){
    int startup_time = 10000*N; // sufficient for rho = .95 at least 
    simulate(startup_time);
    double *average;
    if ( only_print_total_number_of_jobs )
      {
	average = new double[1];
	average[0]=0;
      }
    else {
      average = new double[20];
      for(int i=0;i<20;i++)
	average[i] = 0;
    }
    int nb_samples = 20000;
    if ( only_print_total_number_of_jobs )
      {
	for(int t=0;t<10*nb_samples;t++){
	  //simulate_twoChoice_withReplacement(10);
	  simulate(10);
	  average[0] += total_number_jobs / (N*10.*nb_samples);
	}
      }
    else {
      for(int t=0;t<nb_samples;t++){
	simulate(100);
	for(int i=0;i<20;i++) {average[i] += X[i]/nb_samples;}
      }
    }
    if (only_print_total_number_of_jobs){
      std::cout << average[0] << "\n";
    }
    else {
      for(int i=0;i<20;i++) std::cout << average[i] << " ";
      std::cout << std::endl;
    }
    delete average;
  }
  void test_convergence(){
    simulate(1000*N); // seems to suffices for rho = 0.99
    for(int t=0;t<10000;t++){
      simulate(1000);
      print();
    }
  }
  void print_one_trajectory(){
    X[1]=1; X[2] = 1;
    total_number_jobs = 2*N;
    for(int t=0;t<10000*N;t++){
      simulate(1);
      print();
    }
  }
};


int main(int argc, char ** argv) {
  int N=1000;
  double rho=0.8;
  int d=2; 
  int nb_experiments = -1;
  bool only_print_one_trajectory = false;
  bool withoutReplacement = false;
  bool only_print_total_number_of_jobs = false; 
  for(int i=1;i<argc;i++){
    switch(* (argv[i]) ){
    case 'N': N=atoi(argv[i]+1); break;
    case 'r': rho=atof(argv[i]+1); break;
    case 'd': d=atoi(argv[i]+1); break;
    case 'e': nb_experiments=atoi(argv[i]+1); break;
    case 'c': {JSQd simu(N,rho,40,d,withoutReplacement,only_print_one_trajectory);
	simu.test_convergence(); exit(1);} break;
    case 't': only_print_one_trajectory=true; break;
    case 'T': only_print_total_number_of_jobs=true; break;
    case 'W': withoutReplacement = true; break;
    }
  }
  time_t t;
  time(&t);
  mt_rand.seed(t+N);
  //mt_rand.seed(2);

  if (only_print_one_trajectory){
    JSQd simu(N,rho,30,d, withoutReplacement,only_print_total_number_of_jobs);
    simu.print_one_trajectory();
  }
  else
    {
      if (nb_experiments<0) nb_experiments = 5+(50 + N*N)/100;
      //std::cerr << N << " " << nb_experiments << "\n";
      for(int i=0;i< nb_experiments;i++) {
	JSQd simu(N,rho,40,d, withoutReplacement,only_print_total_number_of_jobs);
	simu.steady_state();
      }
    }
}
