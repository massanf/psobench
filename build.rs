fn main() {
  cc::Build::new().file("src/cec17/cec17_test_func.c").include("src/cec17").compile("libcec17.a");
}
